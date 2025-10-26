//
//  AudioStreamingClient.swift
//  janusai
//
//  Updated to use HTTP streaming instead of WebSockets
//

import Foundation
import AVFoundation
import Combine

final class AudioStreamingClient: NSObject, ObservableObject {
    enum State { case idle, connecting, running, stopped, failed(Error) }

    @Published var state: State = .idle
    @Published var rmsLevel: Float = 0

    private var urlSession: URLSession!
    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let varispeed = AVAudioUnitVarispeed()
    private var audioFormat24k: AVAudioFormat!
    private var tapInstalled: Bool = false
    
    // HTTP streaming tasks
    private var uploadTask: URLSessionTask?
    private var downloadTask: URLSessionDataTask?
    private var sessionId: String = ""
    private var settingsCancellable: AnyCancellable?
    
    // Upload queue for captured audio
    private let uploadQueue = DispatchQueue(label: "AudioStreamingClient.upload")
    private var pendingUploadChunks: [Data] = []
    private var isUploading: Bool = false
    
    // Debug counters
    private var captureCount = 0
    private var uploadCount = 0
    private var downloadBytesReceived = 0

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 3600  // 1 hour timeout for streaming
        config.timeoutIntervalForResource = 3600
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        audioFormat24k = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24_000, channels: 1, interleaved: true)
        audioEngine.attach(playerNode)
        audioEngine.attach(varispeed)
        audioEngine.connect(playerNode, to: varispeed, format: audioFormat24k)
        audioEngine.connect(varispeed, to: audioEngine.mainMixerNode, format: audioFormat24k)
    }

    func start(sessionId: String, webSocketURL url: URL) {
        // Note: webSocketURL parameter kept for API compatibility but unused
        self.start(sessionId: sessionId)
    }
    
    func start(sessionId: String) {
        if case .connecting = state { return }
        if case .running = state { return }
        
        self.sessionId = sessionId
        state = .connecting
        
        // Reset counters
        captureCount = 0
        uploadCount = 0
        downloadBytesReceived = 0
        
        prepareAudioSession { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.state = .failed(NSError(domain: "AudioStreamingClient", code: 1, userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"]))
                return
            }
            
            // Start download stream (receive audio from server)
            self.startDownloadStream()
            
            // Start audio capture and upload
            self.startPlayer()
            self.startCapture()
            
            DispatchQueue.main.async {
                print("🟢 State changed to: running")
                self.state = .running
            }
        }
    }

    func stop() {
        // Stop upload task
        uploadTask?.cancel()
        uploadTask = nil
        
        // Stop download stream
        downloadTask?.cancel()
        downloadTask = nil
        
        // Stop audio stream
        if let url = APIService.shared.stopStreamURL(sessionId: sessionId) {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            urlSession.dataTask(with: request).resume()
        }
        
        // Stop audio capture
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.stop()
        playerNode.stop()
        
        // Reset upload queue
        uploadQueue.async {
            self.pendingUploadChunks.removeAll()
            self.isUploading = false
        }
        
        state = .stopped
    }

    // MARK: - Download Stream (Server → Client)
    private func startDownloadStream() {
        guard let url = APIService.shared.streamAudioURL(sessionId: sessionId) else {
            print("❌ Failed to construct stream audio URL")
            return
        }
        
        print("📥 Starting download stream from: \(url.absoluteString)")
        downloadTask = urlSession.dataTask(with: url)
        downloadTask?.resume()
    }

    // MARK: - Capture & Upload (Client → Server)
    private func startCapture() {
        if tapInstalled { 
            print("⚠️ Tap already installed, skipping")
            return 
        }
        
        self.captureCount = 0
        let bufferSize: AVAudioFrameCount = 1024
        
        // Get the input node BEFORE preparing/starting
        let input: AVAudioInputNode = audioEngine.inputNode
        
        // Important: Prepare and start the engine BEFORE installing tap
        // This ensures the input node's format is properly initialized
        audioEngine.prepare()
        print("✅ Audio engine prepared")
        
        do {
            try audioEngine.start()
            print("✅ Audio engine started")
        } catch {
            print("❌ Failed to start audio engine: \(error)")
            state = .failed(error)
            return
        }
        
        // Now start the player node since engine is running
        playerNode.play()
        print("✅ Player node started")
        
        // Wait a moment for hardware to initialize, then install tap
        // This is necessary because the input format may not be immediately available
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            
            // Get the actual hardware input format (after engine is started and hardware initialized)
            let inputFormat = input.outputFormat(forBus: 0)
            print("🎙️ Input hardware format: \(inputFormat)")
            
            // Validate format before installing tap
            guard inputFormat.sampleRate > 0 && inputFormat.channelCount > 0 else {
                print("❌ Invalid input format: sample rate = \(inputFormat.sampleRate), channels = \(inputFormat.channelCount)")
                self.state = .failed(NSError(domain: "AudioStreamingClient", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid audio input format"]))
                return
            }
            
            print("📍 Installing tap on input node...")
            input.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
                guard let self else { return }
                self.captureCount += 1
                
                if self.captureCount == 1 {
                    print("🎙️ Actual capture format: \(buffer.format)")
                }
                
                if self.captureCount <= 3 || self.captureCount % 100 == 0 {
                    print("🎤 Captured buffer #\(self.captureCount), frames: \(buffer.frameLength)")
                }
                
                if let data = Self.convertToPCM16Mono(buffer: buffer, sampleRate: 24_000) {
                    self.updateRMS(fromPCM16: data)
                    self.enqueueUpload(data)
                }
            }
            print("✅ Tap installed successfully - recording will begin")
            self.tapInstalled = true
        }
    }

    private func startPlayer() {
        // Kept for API compatibility
    }

    // MARK: - Upload Queue
    private func enqueueUpload(_ data: Data) {
        uploadQueue.async {
            // Keep queue bounded (~2 seconds at 40ms/packet)
            let maxQueued = 50
            if self.pendingUploadChunks.count >= maxQueued {
                self.pendingUploadChunks.removeFirst(self.pendingUploadChunks.count - maxQueued + 1)
            }
            self.pendingUploadChunks.append(data)
            self.flushUploadQueue()
        }
    }

    private func flushUploadQueue() {
        uploadQueue.async {
            guard !self.isUploading, !self.pendingUploadChunks.isEmpty else { return }
            self.isUploading = true
            let chunk = self.pendingUploadChunks.removeFirst()
            self.uploadCount += 1
            
            if self.uploadCount <= 3 || self.uploadCount % 100 == 0 {
                print("📤 Uploading chunk #\(self.uploadCount), size: \(chunk.count) bytes, queue: \(self.pendingUploadChunks.count)")
            }
            
            self.uploadAudioChunk(chunk) { [weak self] success in
                guard let self else { return }
                self.uploadQueue.async {
                    self.isUploading = false
                    if !success {
                        // On error, clear queue to prevent memory growth
                        self.pendingUploadChunks.removeAll()
                    }
                    self.flushUploadQueue()
                }
            }
        }
    }

    private func uploadAudioChunk(_ data: Data, completion: @escaping (Bool) -> Void) {
        guard let url = APIService.shared.uploadAudioChunkURL(sessionId: sessionId) else {
            completion(false)
            return
        }
        
        // Prepend timestamp (milliseconds since epoch as Int64)
        let timestampMs = Int64(Date().timeIntervalSince1970 * 1000)
        var packetData = Data()
        withUnsafeBytes(of: timestampMs.littleEndian) { bytes in
            packetData.append(contentsOf: bytes)
        }
        packetData.append(data)
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        
        let task = urlSession.uploadTask(with: request, from: packetData) { data, response, error in
            if let error = error {
                print("❌ Upload error: \(error.localizedDescription)")
                completion(false)
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 200 {
                    completion(true)
                } else {
                    print("❌ Upload failed with status: \(httpResponse.statusCode)")
                    completion(false)
                }
            } else {
                completion(false)
            }
        }
        task.resume()
    }

    // MARK: - Playback
    private func schedulePlayback(pcm16: Data) {
        let frameCount = UInt32(pcm16.count / 2)
        guard let buf = AVAudioPCMBuffer(pcmFormat: audioFormat24k, frameCapacity: frameCount) else { return }
        buf.frameLength = frameCount
        pcm16.withUnsafeBytes { raw in
            guard let srcBase = raw.baseAddress else { return }
            // Copy into channel 0 (non-interleaved)
            memcpy(srcBuf.int16ChannelData![0], srcBase, pcm16.count)
        }

        // Determine destination (player node) format
        let dstFormat = playerNode.outputFormat(forBus: 0)
        let ratio = dstFormat.sampleRate / sourcePCM16_24k.sampleRate
        let dstCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * ratio + 16)
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstCapacity) else { return }

        guard let converter = AVAudioConverter(from: sourcePCM16_24k, to: dstFormat) else { return }
        var error: NSError?
        var consumedSrc = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if consumedSrc {
                outStatus.pointee = .noDataNow
                return nil
            }
            outStatus.pointee = .haveData
            consumedSrc = true
            return srcBuf
        }
        let status = converter.convert(to: dstBuf, error: &error, withInputFrom: inputBlock)
        guard status != .error, error == nil else { return }
        playerNode.scheduleBuffer(dstBuf, completionHandler: nil)
    }

    // MARK: - Utils
    private func updateRMS(fromPCM16 data: Data) {
        let count = data.count / 2
        var sumSquares: Float = 0
        data.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Int16.self).baseAddress!
            for i in 0..<count {
                let v = Float(ptr[i]) / 32768.0
                sumSquares += v * v
            }
        }
        let rms = sqrtf(sumSquares / Float(max(count, 1)))
        DispatchQueue.main.async { self.rmsLevel = rms }
    }

    static func convertToPCM16Mono(buffer: AVAudioPCMBuffer, sampleRate: Double) -> Data? {
        let inputFormat = buffer.format
        guard let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                                 sampleRate: sampleRate,
                                                 channels: 1,
                                                 interleaved: true) else {
            return nil
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: desiredFormat) else {
            return nil
        }
        let ratio = desiredFormat.sampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: desiredFormat, frameCapacity: outCapacity) else {
            return nil
        }
        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        let status = converter.convert(to: outBuf, error: &error, withInputFrom: inputBlock)
        guard status != .error, error == nil, let channelData = outBuf.int16ChannelData else {
            return nil
        }
        let frameCount = Int(outBuf.frameLength)
        let bytes = UnsafeBufferPointer(start: channelData[0], count: frameCount)
        return Data(buffer: bytes)
    }
}

// MARK: - URLSessionDataDelegate
extension AudioStreamingClient: URLSessionDataDelegate {
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        // Received audio data from server stream
        downloadBytesReceived += data.count
        
        if downloadBytesReceived <= 1000 || downloadBytesReceived % 10000 < 100 {
            print("📥 Received \(data.count) bytes from stream (total: \(downloadBytesReceived))")
        }
        
        // Skip WAV header (first 44 bytes)
        let audioData: Data
        if downloadBytesReceived <= 44 {
            // Skip header
            return
        } else if downloadBytesReceived - data.count < 44 {
            // This chunk contains part of the header, skip it
            let headerRemaining = 44 - (downloadBytesReceived - data.count)
            audioData = data.dropFirst(headerRemaining)
        } else {
            audioData = data
        }
        
        // Schedule for playback
        if !audioData.isEmpty {
            schedulePlayback(pcm16: audioData)
        }
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("❌ Stream error: \(error.localizedDescription)")
            DispatchQueue.main.async {
                if case .running = self.state {
                    self.state = .failed(error)
                }
            }
        } else {
            print("✅ Stream completed")
        }
    }
}

// MARK: - Audio session
private extension AudioStreamingClient {
    func prepareAudioSession(completion: @escaping (Bool) -> Void) {
        print("🔧 Preparing audio session...")
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setPreferredSampleRate(24_000)
            try session.setPreferredIOBufferDuration(0.02)
            print("✅ Audio session configured")
        } catch {
            print("❌ Failed to configure audio session: \(error)")
            completion(false)
            return
        }
        session.requestRecordPermission { granted in
            print("🎤 Microphone permission: \(granted ? "GRANTED" : "DENIED")")
            if granted {
                do { 
                    try session.setActive(true) 
                    print("✅ Audio session activated")
                } catch {
                    print("❌ Failed to activate audio session: \(error)")
                }
            }
            DispatchQueue.main.async { completion(granted) }
        }
    }
}

// MARK: - Settings-driven playback speed
extension AudioStreamingClient {
    func bindSettings(_ publisher: AnyPublisher<APIService.Settings, Never>) {
        settingsCancellable?.cancel()
        settingsCancellable = publisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] settings in
                guard let self else { return }
                // Map playbackSpeed -> varispeed.rate (1.0 = normal)
                let rate: Float
                switch settings.playbackSpeed.lowercased() {
                case "slow": rate = 0.85
                case "fast": rate = 1.25
                default: rate = 1.0
                }
                self.varispeed.rate = rate
                print("🎛️ Varispeed rate set to \(rate) for speed=\(settings.playbackSpeed)")
            }
    }
}
