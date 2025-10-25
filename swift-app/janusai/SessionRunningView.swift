//
//  SessionRunningView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI
import AVFoundation
import Combine

final class StreamAudioPlayer: ObservableObject {
    @Published var samples: [Float] = Array(repeating: 0, count: 120)
    private var audioEngine: AVAudioEngine?
    private var playerNode = AVAudioPlayerNode()
    private var timer: Timer?

    func start() {
        let engine = AVAudioEngine()
        audioEngine = engine
        engine.attach(playerNode)
        let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)

        do {
            try engine.start()
            playerNode.play()
        } catch { print("engine start error: \(error)") }

        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self else { return }
            // Placeholder: animate waveform until real levels from stream
            self.samples.removeFirst()
            self.samples.append(Float.random(in: 0.05...0.9))
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        playerNode.stop()
        audioEngine?.stop()
        audioEngine = nil
    }
}

struct SessionRunningView: View {
    @StateObject private var player = StreamAudioPlayer()

    var body: some View {
        VStack(spacing: 16) {
            // Device switcher
            RoutePicker()
                .frame(width: 44, height: 44)
                .padding(.top, 8)

            // Edge-to-edge waveform
            WaveformView(samples: player.samples)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)

            HStack {
                Button("Stop") { player.stop() }
                    .buttonStyle(.bordered)
                Button("Start") { player.start() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .navigationTitle("Session Running")
        .interactiveDismissDisabled(true)
        .onAppear { player.start() }
        .onDisappear { player.stop() }
    }
}

