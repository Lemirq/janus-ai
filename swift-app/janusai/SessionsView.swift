//
//  SessionsView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI

struct SessionsView: View {
    @State private var queryText: String = ""
    @State private var results: [APIService.QueryResult] = []
    @State private var status: String = ""
    @State private var isLoading = false

    var body: some View {
        NavigationView {
            VStack(spacing: 12) {
            

                if !status.isEmpty { Text(status).font(.footnote).foregroundColor(.secondary) }

                if results.isEmpty {
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
                    List(results, id: \.id) { item in
                        VStack(alignment: .leading) {
                            Text(item.text).font(.body)
                            if let d = item.distance {
                                Text(String(format: "Distance: %.3f", d)).font(.caption).foregroundColor(.secondary)
                            }
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
        }
    }

    private func doQuery() async {
        isLoading = true
        defer { isLoading = false }
        do {
            results = try await APIService.shared.queryDocuments(query: queryText)
            status = "Found \(results.count) results"
        } catch {
            status = "Query failed: \(error.localizedDescription)"
        }
    }
}


