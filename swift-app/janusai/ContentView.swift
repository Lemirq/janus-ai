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
            ZStack {
                LinearGradient(colors: [Color.blue.opacity(0.35), Color.purple.opacity(0.35)], startPoint: .topLeading, endPoint: .bottomTrailing)
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 20) {
                        VStack(spacing: 12) {
                            Image(systemName: "sparkles")
                                .font(.system(size: 42, weight: .semibold))
                                .foregroundColor(.white)
                                .shadow(radius: 6)
                            Text("Welcome to Janus AI")
                                .font(.largeTitle.weight(.bold))
                                .foregroundColor(.white)
                            Text("Get set up in a few quick steps.")
                                .foregroundColor(.white.opacity(0.9))
                        }
                        .padding(.top, 20)

                        VStack(alignment: .leading, spacing: 16) {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Label("Backend Health", systemImage: "heart.text.square")
                                        .font(.headline)
                                    Spacer()
                                    healthBadge
                                }
                                HStack(spacing: 10) {
                                    if health == "Checking..." {
                                        ProgressView().progressViewStyle(.circular)
                                    }
                                    Button("Re-check") { Task { await check() } }
                                        .buttonStyle(.bordered)
                                }
                            }

                            Divider()

                            VStack(alignment: .leading, spacing: 10) {
                                HStack(spacing: 10) {
                                    Image(systemName: "mic.fill")
                                        .foregroundColor(.accentColor)
                                    Text("Voice Sample")
                                        .font(.headline)
                                }
                                Text("Record a short sample to help personalize the voice to you")
                                    .foregroundColor(.secondary)
                                NavigationLink(destination: VoiceSetupView(onComplete: { completedOnboarding = true })) {
                                    Text("Record Sample")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .padding(20)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: 6)

                        EmptyView()
                    }
                    .padding()
                }
            }
            .navigationTitle("Onboarding")
            .navigationBarTitleDisplayMode(.inline)
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
extension OnboardingView {
    private var healthBadge: some View {
        let (text, color, symbol) = healthStyle()
        return HStack(spacing: 6) {
            Image(systemName: symbol)
            Text(text)
                .font(.subheadline.weight(.semibold))
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(color.opacity(0.15))
        .foregroundColor(color)
        .clipShape(Capsule())
    }

    private func healthStyle() -> (String, Color, String) {
        if health == "OK" { return ("OK", .green, "checkmark.circle.fill") }
        if health == "Down" { return ("Down", .red, "xmark.octagon.fill") }
        if health.hasPrefix("Error") { return ("Error", .orange, "exclamationmark.triangle.fill") }
        return ("Checking...", .yellow, "clock.fill")
    }
}

