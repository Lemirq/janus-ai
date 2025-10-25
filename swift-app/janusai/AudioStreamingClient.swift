//
//  AudioStreamingClient.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import Foundation
import AVFoundation
import Combine

final class AudioStreamingClient: NSObject, ObservableObject {
    enum State { case idle, connecting, running, stopped, failed(Error) }

    @Published var state: State = .idle
    @Published var rmsLevel: Float = 0

    private var webSocket: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var audioFormat16k: AVAudioFormat!
    private var tapInstalled: Bool = false

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        audioFormat16k = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat16k)
    }

    func start(sessionId: String) {
        guard let url = APIService.shared.webSocketURL(sessionId: sessionId) else { return }
        if case .connecting = state { return }
        if case .running = state { return }
        state = .connecting
        webSocket = urlSession.webSocketTask(with: url)
        webSocket?.resume()
        receiveLoop()
        startCapture()
        startPlayer()
        sendStartControl()
    }

    func stop() {
        sendStopControl()
        webSocket?.cancel(with: .normalClosure, reason: nil)
        webSocket = nil
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.stop()
        playerNode.stop()
        state = .stopped
    }

    // MARK: - Capture
    private func startCapture() {
        if tapInstalled { return }
        let input = audioEngine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        let bufferSize: AVAudioFrameCount = 1024
        input.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            if let data = Self.convertToPCM16Mono16k(buffer: buffer) {
                self.updateRMS(fromPCM16: data)
                self.sendBinary(data)
            }
        }
        tapInstalled = true
        do { try audioEngine.start() } catch { state = .failed(error) }
    }

    private func startPlayer() {
        if !audioEngine.isRunning {
            do { try audioEngine.start() } catch { state = .failed(error) }
        }
        playerNode.play()
    }

    // MARK: - WebSocket
    private func receiveLoop() {
        webSocket?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                DispatchQueue.main.async { self.state = .failed(error) }
            case .success(let message):
                switch message {
                case .data(let data):
                    self.schedulePlayback(pcm16: data)
                case .string:
                    break
                @unknown default:
                    break
                }
                self.receiveLoop()
            }
        }
    }

    private func sendBinary(_ data: Data) {
        webSocket?.send(.data(data)) { _ in }
    }

    private func sendStartControl() {
        let obj: [String: Any] = ["type": "start", "sampleRate": 16_000]
        if let data = try? JSONSerialization.data(withJSONObject: obj), let str = String(data: data, encoding: .utf8) {
            webSocket?.send(.string(str)) { _ in }
        }
    }

    private func sendStopControl() {
        let obj: [String: Any] = ["type": "stop"]
        if let data = try? JSONSerialization.data(withJSONObject: obj), let str = String(data: data, encoding: .utf8) {
            webSocket?.send(.string(str)) { _ in }
        }
    }

    // MARK: - Playback
    private func schedulePlayback(pcm16: Data) {
        let frameCount = UInt32(pcm16.count / 2)
        guard let buf = AVAudioPCMBuffer(pcmFormat: audioFormat16k, frameCapacity: frameCount) else { return }
        buf.frameLength = frameCount
        pcm16.withUnsafeBytes { raw in
            guard let src = raw.baseAddress else { return }
            memcpy(buf.int16ChannelData![0], src, pcm16.count)
        }
        playerNode.scheduleBuffer(buf, completionHandler: nil)
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

    static func convertToPCM16Mono16k(buffer: AVAudioPCMBuffer) -> Data? {
        let inputFormat = buffer.format
        guard let desired = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true) else { return nil }
        let converter = AVAudioConverter(from: inputFormat, to: desired)
        let outCapacity = AVAudioFrameCount(640) // ~40ms@16k
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: desired, frameCapacity: outCapacity) else { return nil }
        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        converter?.convert(to: outBuf, error: &error, withInputFrom: inputBlock)
        if let error { print("convert error: \(error)"); return nil }
        guard let ch = outBuf.int16ChannelData else { return nil }
        let frames = Int(outBuf.frameLength)
        let bytes = UnsafeBufferPointer(start: ch[0], count: frames)
        return Data(buffer: bytes)
    }
}

extension AudioStreamingClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        DispatchQueue.main.async { self.state = .running }
    }
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        DispatchQueue.main.async { self.state = .stopped }
    }
}


