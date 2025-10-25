//
//  SettingsView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("webSearchEnabled") private var webSearchEnabled = false
    @AppStorage("languageCode") private var languageCode = "en"

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
        }
        .navigationTitle("Settings")
    }
}


