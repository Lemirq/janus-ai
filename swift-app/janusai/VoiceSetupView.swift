//
//  VoiceSetupView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI
import AVFoundation

struct VoiceSetupView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var isRecording = false
    @State private var isPlaying = false
    @State private var status: String = ""
    @State private var recordingURL: URL?
    @State private var startTime: Date?
    @State private var audioPlayer: AVAudioPlayer?

    private let presetSentence = "The quick brown fox jumps over the lazy dog."
    var onComplete: (() -> Void)? = nil

    // AV
    private let audioEngine = AVAudioEngine()
    private var inputNode: AVAudioInputNode? { audioEngine.inputNode }

    var body: some View {
        VStack(spacing: 16) {
            Text("Please read this sentence out loud at your usual pace:")
            Text(presetSentence).font(.headline).multilineTextAlignment(.center)

            if let url = recordingURL, !isRecording {
                VStack(spacing: 8) {
                    HStack(spacing: 16) {
                        Button(isPlaying ? "Pause" : "Play") { togglePlay(url: url) }
                            .buttonStyle(.bordered)
                        Button("Re-record") { Task { await rerecord() } }
                            .buttonStyle(.bordered)
                        Button("Continue") { Task { await uploadRecording() } }
                            .buttonStyle(.borderedProminent)
                    }
                }
            } else {
                Button(isRecording ? "Stop" : "Record") { Task { await toggleRecording() } }
                    .buttonStyle(.borderedProminent)
            }

            if !status.isEmpty {
                Text(status).font(.footnote).foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding()
        .navigationTitle("Voice Setup")
        .task { await requestMicPermissionIfNeeded() }
    }
}

// MARK: - Recording / Playback
extension VoiceSetupView {
    private func requestMicPermissionIfNeeded() async {
        let granted = await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in
            AVAudioSession.sharedInstance().requestRecordPermission { c.resume(returning: $0) }
        }
        if !granted { status = "Microphone permission denied" }
    }

    private func toggleRecording() async {
        if isRecording { await stopRecording(); return }
        await startRecording()
    }

    @MainActor private func startRecording() async {
        do {
            status = "Recording..."
            startTime = Date()

            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .duckOthers])
            try session.setActive(true)

            guard let inputNode = inputNode else {
                status = "No audio input available"
                return
            }
            let format = inputNode.outputFormat(forBus: 0)

            // Setup file
            let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("recording.caf")
            if FileManager.default.fileExists(atPath: tmp.path) { try? FileManager.default.removeItem(at: tmp) }
            let file = try AVAudioFile(forWriting: tmp, settings: format.settings)
            recordingURL = tmp

            // Tap audio to write to file
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                try? file.write(from: buffer)
            }

            audioEngine.prepare()
            try audioEngine.start()

            isRecording = true
        } catch {
            status = "Failed to start recording: \(error.localizedDescription)"
        }
    }

    @MainActor private func stopRecording() async {
        audioEngine.stop()
        inputNode?.removeTap(onBus: 0)
        isRecording = false
        status = "Recording stopped"
    }

    @MainActor private func uploadRecording() async {
        guard let url = recordingURL else { return }
        if isRecording { await stopRecording() }
        let duration = max(0, audioDurationSeconds(of: url) ?? -(startTime?.timeIntervalSinceNow ?? 0))
        do {
            let resp = try await APIService.shared.uploadVoice(fileURL: url, transcript: "", duration: duration)
            status = "Uploaded. Estimated WPM: \(resp.wpm ?? 0)"
            onComplete?()
            dismiss()
        } catch {
            status = "Upload failed: \(error.localizedDescription)"
        }
    }

    private func togglePlay(url: URL) {
        if isPlaying {
            audioPlayer?.pause()
            isPlaying = false
            return
        }
        do {
            // Ensure playback routes to the bottom loudspeaker
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .duckOthers])
            try session.overrideOutputAudioPort(.speaker)
            try session.setActive(true)

            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            isPlaying = true
        } catch {
            status = "Playback failed: \(error.localizedDescription)"
        }
    }

    private func rerecord() async {
        if isRecording { await stopRecording() }
        if let url = recordingURL { try? FileManager.default.removeItem(at: url) }
        recordingURL = nil
        status = ""
        isPlaying = false
    }

    private func audioDurationSeconds(of url: URL) -> TimeInterval? {
        let asset = AVURLAsset(url: url)
        return CMTimeGetSeconds(asset.duration)
    }
}
