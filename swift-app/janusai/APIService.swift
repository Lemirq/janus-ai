//
//  APIService.swift
//  janusai
//
//  Created by Vihaan on 2025-10-25.
//

import Foundation

final class APIService {
    static let shared = APIService()

    // Update for production environments as needed
    private let baseURL = "http://localhost:2025/api"

    private let jsonDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        return decoder
    }()

    private let jsonEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        return encoder
    }()

    // MARK: - Health
    func checkHealth() async throws -> Bool {
        guard let url = URL(string: "\(baseURL)/health") else { return false }
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try jsonDecoder.decode([String: String].self, from: data)
        return response["status"] == "ok"
    }

    // MARK: - Ingest
    struct IngestResponse: Codable { let added: Int }

    func ingestDocuments(_ documents: [String], collection: String? = nil) async throws -> Int {
        guard let url = URL(string: "\(baseURL)/ingest") else { throw URLError(.badURL) }

        var body: [String: Any] = ["documents": documents]
        if let collection { body["collection"] = collection }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try jsonDecoder.decode(IngestResponse.self, from: data)
        return response.added
    }

    // MARK: - Query
    struct QueryResponse: Codable { let results: [QueryResult] }

    struct QueryResult: Codable {
        let id: String
        let text: String
        let metadata: [String: String]? // Backend currently returns null; broaden type if needed later
        let distance: Double?
    }

    func queryDocuments(query: String, topK: Int = 5, collection: String? = nil) async throws -> [QueryResult] {
        var components = URLComponents(string: "\(baseURL)/query")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "top_k", value: String(topK)),
        ]
        if let collection { items.append(URLQueryItem(name: "collection", value: collection)) }
        components.queryItems = items

        guard let url = components.url else { throw URLError(.badURL) }
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try jsonDecoder.decode(QueryResponse.self, from: data)
        return response.results
    }
}

// MARK: - Voice Upload
extension APIService {
    struct VoiceUploadResponse: Codable { let id: String; let wpm: Double?; let duration: Double; let words: Int }

    func uploadVoice(fileURL: URL, transcript: String, duration: TimeInterval) async throws -> VoiceUploadResponse {
        guard let url = URL(string: "\(baseURL)/voice/upload") else { throw URLError(.badURL) }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func appendField(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }
        func appendFileField(name: String, filename: String, mime: String, fileData: Data) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: \(mime)\r\n\r\n".data(using: .utf8)!)
            body.append(fileData)
            body.append("\r\n".data(using: .utf8)!)
        }

        let fileData = try Data(contentsOf: fileURL)
        appendFileField(name: "file", filename: fileURL.lastPathComponent, mime: "audio/caf", fileData: fileData)
        appendField(name: "transcript", value: transcript)
        appendField(name: "duration", value: String(duration))
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        let (data, _) = try await URLSession.shared.upload(for: request, from: body)
        return try jsonDecoder.decode(VoiceUploadResponse.self, from: data)
    }
}