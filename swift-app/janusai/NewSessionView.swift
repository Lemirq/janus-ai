//
//  NewSessionView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI

struct NewSessionView: View {
    @State private var objective: String = ""
    @State private var pickedDocs: [PickedDocument] = []
    @State private var isPicking = false
    @State private var isSubmitting = false
    @State private var status: String = ""
    @State private var navigateToRun = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Objective description")
                    .font(.headline)
                TextField("Describe your objective...", text: $objective, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)

                Divider()

                Text("Upload files (ingest to Chroma)")
                    .font(.headline)
                if pickedDocs.isEmpty {
                    Text("No files selected")
                        .foregroundColor(.secondary)
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(pickedDocs) { doc in
                            Text(doc.url.lastPathComponent)
                                .font(.subheadline)
                                .lineLimit(1)
                        }
                    }
                }

                HStack {
                    Button("Choose Files") { isPicking = true }
                    Button("Ingest") { Task { await ingest() } }
                        .disabled(pickedDocs.isEmpty || isSubmitting)
                }

                Divider()

                NavigationLink(destination: SessionRunningView(), isActive: $navigateToRun) { EmptyView() }
                    .hidden()
                Button("Start Session") {
                    Task { await startSession() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(objective.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSubmitting)

                if !status.isEmpty {
                    Text(status).font(.footnote).foregroundColor(.secondary)
                }

                Spacer(minLength: 20)
            }
            .padding()
        }
        .sheet(isPresented: $isPicking) {
            DocumentPicker { docs in
                pickedDocs = docs
            } onCancel: {}
        }
        .navigationTitle("New Session")
    }

    private func ingest() async {
        isSubmitting = true
        defer { isSubmitting = false }
        do {
            // Basic approach: send raw text chunks
            let texts = pickedDocs.map { $0.text }
            let added = try await APIService.shared.ingestDocuments(texts)
            status = "Ingested \(added) documents"
        } catch {
            status = "Ingest failed: \(error.localizedDescription)"
        }
    }

    private func startSession() async {
        // Placeholder: In a real app, call backend to create/start a session with objective + resources
        // For now, simply navigate to running screen when we have at least an objective.
        navigateToRun = true
    }
}


