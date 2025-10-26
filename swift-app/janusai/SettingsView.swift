//
//  SettingsView.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-25.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("webSearchEnabled") private var webSearchEnabled = false
    @AppStorage("languageCode") private var languageCode = "en"
    @AppStorage("completedOnboarding") private var completedOnboarding = false
    @State private var showClearConfirm = false
    @State private var devStatus: String = ""

    private let languages: [(code: String, name: String)] = [
        ("en", "English"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("hi", "Hindi"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
    ]

    var body: some View {
        Form {
            Section(header: Text("Web Search")) {
                Toggle("Enable Web Search", isOn: $webSearchEnabled)
            }
            Section(header: Text("Language")) {
                Picker("Language", selection: $languageCode) {
                    ForEach(languages, id: \.code) { lang in
                        Text(lang.name).tag(lang.code)
                    }
                }
            }
            Section(header: Text("Developer"), footer: devFooter) {
                Button(role: .destructive) {
                    showClearConfirm = true
                } label: {
                    Label("Clear App Storage", systemImage: "trash")
                }
            }
        }
        .navigationTitle("Settings")
        .alert("Clear App Storage?", isPresented: $showClearConfirm) {
            Button("Cancel", role: .cancel) {}
            Button("Clear", role: .destructive) { clearAppStorage() }
        } message: {
            Text("This will reset all preferences and onboarding. You'll redo voice setup next launch.")
        }
    }
}


extension SettingsView {
    private var devFooter: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Clears all saved preferences and flags.")
                .foregroundColor(.secondary)
                .font(.footnote)
            if !devStatus.isEmpty {
                Text(devStatus)
                    .foregroundColor(.secondary)
                    .font(.footnote)
            }
        }
    }

    private func clearAppStorage() {
        if let bundleId = Bundle.main.bundleIdentifier {
            UserDefaults.standard.removePersistentDomain(forName: bundleId)
            UserDefaults.standard.synchronize()
        }
        completedOnboarding = false
        devStatus = "Cleared. Restart the app to re-run onboarding."
    }
}


