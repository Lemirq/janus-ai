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
    private var timer: Timer?
    private var client = AudioStreamingClient()
    private var sessionId: String = ""

    func start(sessionId: String) {
        self.sessionId = sessionId
        client.start(sessionId: sessionId)
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.samples.removeFirst()
            self.samples.append(self.client.rmsLevel)
        }
    }

    func stop() {
        timer?.invalidate(); timer = nil
        client.stop()
    }
}

struct SessionRunningView: View {
    let sessionId: String
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
                Button("Start") { player.start(sessionId: sessionId) }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .navigationTitle("Session Running")
        .interactiveDismissDisabled(true)
        .onAppear { player.start(sessionId: sessionId) }
        .onDisappear { player.stop() }
    }
}

