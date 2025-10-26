//
//  SessionsView.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-25.
//

import SwiftUI


struct SessionsView: View {
    @State private var sessions: [APIService.SessionSummary] = []
    @State private var status: String = ""
    @State private var isLoading = false

    var body: some View {
        NavigationView {
            VStack(spacing: 12) {
            

                if !status.isEmpty { Text(status).font(.footnote).foregroundColor(.secondary) }

                if sessions.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "tray")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("No sessions yet")
                            .font(.headline)
                        Text("Tap the + button to create your first session.")
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(sessions, id: \.id) { s in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(s.objective).font(.body)
                                if let created = s.createdAt { Text(created).font(.caption).foregroundColor(.secondary) }
                            }
                            Spacer()
                            Text((s.status ?? "").capitalized)
                                .font(.caption.weight(.semibold))
                                .padding(.vertical, 4)
                                .padding(.horizontal, 8)
                                .background(statusColor(s.status).opacity(0.15))
                                .foregroundColor(statusColor(s.status))
                                .clipShape(Capsule())
                        }
                    }
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                NavigationLink(destination: NewSessionView()) {
                    Image(systemName: "plus.circle.fill")
                }
            }
            .task { await reload() }
            .refreshable { await reload() }
        }
    }

    private func reload() async {
        isLoading = true
        defer { isLoading = false }
        do {
            sessions = try await APIService.shared.listSessions()
            status = "Loaded \(sessions.count) sessions"
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
    }
}

private func statusColor(_ status: String?) -> Color {
    switch (status ?? "").lowercased() {
    case "created": return .gray
    case "running": return .blue
    case "stopped": return .orange
    case "completed": return .green
    default: return .secondary
    }
}


