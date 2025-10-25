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
    private let baseURL = "http://100.67.82.198:2025/api"

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
    struct IngestResponse: Codable { let added: Int; let ids: [String]? }

    func ingestDocuments(_ documents: [String], collection: String? = nil, sessionId: String? = nil) async throws -> IngestResponse {
        guard let url = URL(string: "\(baseURL)/ingest") else { throw URLError(.badURL) }

        var body: [String: Any] = ["documents": documents]
        if let collection { body["collection"] = collection }
        if let sessionId { body["sessionId"] = sessionId }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try jsonDecoder.decode(IngestResponse.self, from: data)
        return response
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
    // Sessions
    struct Session: Codable { let id: String; let objective: String; let fileIds: [String]?; let status: String? }

    func createSession(objective: String, fileIds: [String]) async throws -> Session {
        guard let url = URL(string: "\(baseURL)/sessions") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["objective": objective, "fileIds": fileIds]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await URLSession.shared.data(for: req)
        return try jsonDecoder.decode(Session.self, from: data)
    }

    func startSession(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/sessions/\(id)/start") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        _ = try await URLSession.shared.data(for: req)
    }

    func stopSession(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/sessions/\(id)/stop") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        _ = try await URLSession.shared.data(for: req)
    }
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

    // MARK: - WebSocket URL helper
    func webSocketURL(sessionId: String) -> URL? {
        // Convert http(s) base to ws(s) and strip trailing /api
        var root = baseURL
        if let range = root.range(of: "/api", options: [.backwards]) {
            root.removeSubrange(range)
        }
        let wsBase: String
        if root.hasPrefix("https://") {
            wsBase = root.replacingOccurrences(of: "https://", with: "wss://")
        } else {
            wsBase = root.replacingOccurrences(of: "http://", with: "ws://")
        }
        return URL(string: "\(wsBase)/ws/sessions/\(sessionId)")
    }
}
