//
//  ContentView.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-24.
//

import SwiftUI

struct ContentView: View {
    @AppStorage("completedOnboarding") private var completedOnboarding = false

    var body: some View {
        TabView {
            SessionsView()
                .tabItem {
                    Label("Sessions", systemImage: "text.magnifyingglass")
                }

            NavigationView { SettingsView() }
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }

            NavigationView { VoiceSetupView() }
                .tabItem {
                    Label("Voice", systemImage: "waveform")
                }
        }
        .sheet(isPresented: .constant(!completedOnboarding)) {
            OnboardingView(completedOnboarding: $completedOnboarding)
        }
    }
}

struct OnboardingView: View {
    @Binding var completedOnboarding: Bool
    @State private var health: String = "Checking..."

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                Text("Welcome to Janus AI")
                    .font(.title2)
                Text("We will set up your environment and preferences.")
                    .foregroundColor(.secondary)

                VStack(spacing: 8) {
                    Text("Backend health: \(health)")
                    Button("Re-check") { Task { await check() } }
                }

                NavigationLink(destination: VoiceSetupView()) {
                    Label("Set up voice", systemImage: "mic.fill")
                }
                .buttonStyle(.bordered)

                Button("Finish") { completedOnboarding = true }
                    .buttonStyle(.borderedProminent)
            }
            .padding()
            .navigationTitle("Onboarding")
            .task { await check() }
        }
    }

    private func check() async {
        do {
            let ok = try await APIService.shared.checkHealth()
            health = ok ? "OK" : "Down"
        } catch {
            health = "Error: \(error.localizedDescription)"
        }
    }
}

