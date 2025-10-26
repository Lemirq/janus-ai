//
//  SessionRunningView.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-25.
//

import SwiftUI
import AVFoundation
import Combine

final class StreamAudioPlayer: ObservableObject {
    @Published var samples: [Float] = Array(repeating: 0, count: 120)
    private var timer: Timer?
    private var client = AudioStreamingClient()
    private var sessionId: String = ""

    func apply(settings: APIService.Settings) {
        client.bindSettings(Just(settings).eraseToAnyPublisher())
    }

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
    @StateObject private var sensorManager = SensorManager()
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

            // Live settings summary based on current phone orientation
            SettingsCard(
                speed: sensorManager.currentSpeed,
                speedMultiplier: sensorManager.currentSpeedMultiplier,
                verbose: sensorManager.verboseMode
            )

            // Compact sensor orientation and gesture-derived zones
            VStack(spacing: 8) {
                HStack {
                    Label("Accelerometer", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                        .font(.headline)
                    Spacer()
                }
                HStack {
                    Text(String(format: "X %.3f", sensorManager.accelerometerX))
                        .font(.system(.footnote, design: .monospaced))
                    Spacer()
                    Text(String(format: "Y %.3f", sensorManager.accelerometerY))
                        .font(.system(.footnote, design: .monospaced))
                    Spacer()
                    Text(String(format: "Z %.3f", sensorManager.accelerometerZ))
                        .font(.system(.footnote, design: .monospaced))
                }
                .foregroundColor(.secondary)

                // Derived zones preview (what settings will become after debounce)
                HStack(spacing: 16) {
                    let y = sensorManager.accelerometerY
                    let derivedSpeed: String = (y < -0.33) ? "slow" : ((y > 0.33) ? "fast" : "medium")
                    let z = sensorManager.accelerometerZ
                    let derivedVerbose = (z >= 0)
                    Label("Derived Speed: \(derivedSpeed.capitalized)", systemImage: derivedSpeed == "slow" ? "tortoise.fill" : (derivedSpeed == "fast" ? "hare.fill" : "gauge.medium"))
                        .foregroundColor(derivedSpeed == "slow" ? .blue : (derivedSpeed == "fast" ? .red : .green))
                    Label("Derived Verbose: \(derivedVerbose ? "ON" : "OFF")", systemImage: "text.bubble.fill")
                        .foregroundColor(derivedVerbose ? .green : .gray)
                    Spacer()
                }
                .font(.caption)
            }
            .padding()
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .shadow(color: .black.opacity(0.06), radius: 3, x: 0, y: 1)
        }
        .padding()
        .navigationTitle("Session Running")
        .navigationBarBackButtonHidden(true)
        .interactiveDismissDisabled(true)
        .onDisappear {
            player.stop()
            sensorManager.stopTracking()
        }
        .onAppear {
            // Tie sensors to active session
            sensorManager.currentSessionId = sessionId
            if appState.isSessionRunning {
                sensorManager.startTracking()
            }
        }
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
            // Begin continuous motion tracking while session runs
            sensorManager.currentSessionId = sessionId
            sensorManager.startTracking()
            // Bind playback speed to settings updates
            // Poll settings every few seconds to update speed, or wire to your own publisher if available
            // Simple polling here for demo
            Task.detached { [weak player] in
                while true {
                    guard let player else { break }
                    do {
                        let settings = try await APIService.shared.getSettings()
                        DispatchQueue.main.async {
                            player.client.bindSettings(Just(settings).eraseToAnyPublisher())
                        }
                    } catch {}
                    try? await Task.sleep(nanoseconds: 3_000_000_000) // 3s
                }
            }
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
            sensorManager.stopTracking()
            await MainActor.run {
                appState.stopSession()
            }
            status = "Stopped"
        } catch {
            player.stop()
            sensorManager.stopTracking()
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
            sensorManager.stopTracking()
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

