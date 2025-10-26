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
    @State private var isStarting = false
    @State private var isStopping = false
    @State private var status: String = ""
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState

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
                Button("Stop") { Task { await stopSessionAndStream() } }
                    .buttonStyle(.bordered)
                    .disabled(isStarting || isStopping)
                Button("Start") { Task { await startSessionAndStream() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(isStarting || isStopping)
                Button("Done") { Task { await completeAndExit() } }
                    .buttonStyle(.bordered)
                    .tint(.green)
                    .disabled(isStarting || isStopping)
            }

            if !status.isEmpty {
                Text(status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .navigationTitle("Session Running")
        .navigationBarBackButtonHidden(true)
        .interactiveDismissDisabled(true)
        .onDisappear { player.stop() }
    }
}

private extension SessionRunningView {
    func startSessionAndStream() async {
        isStarting = true
        status = "Starting..."
        defer { isStarting = false }
        do {
            try await APIService.shared.startSession(id: sessionId)
            player.start(sessionId: sessionId)
            await MainActor.run {
                appState.startSession(sessionId)
            }
            status = "Started"
        } catch {
            status = "Start failed: \(error.localizedDescription)"
        }
    }

    func stopSessionAndStream() async {
        isStopping = true
        status = "Stopping..."
        defer { isStopping = false }
        do {
            try await APIService.shared.stopSession(id: sessionId)
            player.stop()
            await MainActor.run {
                appState.stopSession()
            }
            status = "Stopped"
        } catch {
            player.stop()
            await MainActor.run {
                appState.stopSession()
            }
            status = "Stopped (API error): \(error.localizedDescription)"
        }
    }

    func completeAndExit() async {
        isStopping = true
        status = "Completing..."
        defer { isStopping = false }
        do {
            player.stop()
            try await APIService.shared.completeSession(id: sessionId)
            await MainActor.run {
                appState.completeSession()
            }
            status = "Completed"
            dismiss()
        } catch {
            status = "Complete failed: \(error.localizedDescription)"
        }
    }
}

