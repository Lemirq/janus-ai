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
        print("[HTTP REQ][Swift] GET \(url.absoluteString)")
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] GET \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        let r = try jsonDecoder.decode([String: String].self, from: data)
        return r["status"] == "ok"
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

        print("[HTTP REQ][Swift] POST \(url.absoluteString) body=\(String(data: request.httpBody ?? Data(), encoding: .utf8) ?? "<binary>")")
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        let r = try jsonDecoder.decode(IngestResponse.self, from: data)
        return r
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
        print("[HTTP REQ][Swift] GET \(url.absoluteString)")
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] GET \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        let r = try jsonDecoder.decode(QueryResponse.self, from: data)
        return r.results
    }
}

// MARK: - Voice Upload
extension APIService {
    // Sessions
    struct Session: Codable { let id: String; let objective: String; let fileIds: [String]?; let status: String?; let createdAt: String? }

    func createSession(objective: String, fileIds: [String]) async throws -> Session {
        guard let url = URL(string: "\(baseURL)/sessions") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["objective": objective, "fileIds": fileIds]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        print("[HTTP REQ][Swift] POST \(url.absoluteString) body=\(String(data: req.httpBody ?? Data(), encoding: .utf8) ?? "<binary>")")
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        return try jsonDecoder.decode(Session.self, from: data)
    }

    func startSession(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/sessions/\(id)/start") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        print("[HTTP REQ][Swift] POST \(url.absoluteString)")
        let (_, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode)")
        }
    }

    func stopSession(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/sessions/\(id)/stop") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        print("[HTTP REQ][Swift] POST \(url.absoluteString)")
        let (_, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode)")
        }
    }

    func completeSession(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/sessions/\(id)/complete") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        print("[HTTP REQ][Swift] POST \(url.absoluteString)")
        let (_, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode)")
        }
    }

    struct SessionsIndex: Codable { let sessions: [SessionSummary] }
    struct SessionSummary: Codable { let id: String; let objective: String; let createdAt: String?; let status: String? }

    func listSessions() async throws -> [SessionSummary] {
        guard let url = URL(string: "\(baseURL)/sessions") else { throw URLError(.badURL) }
        print("[HTTP REQ][Swift] GET \(url.absoluteString)")
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] GET \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        let idx = try jsonDecoder.decode(SessionsIndex.self, from: data)
        return idx.sessions
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

        print("[HTTP REQ][Swift] POST \(url.absoluteString) multipart bytes=\(body.count)")
        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        if let http = response as? HTTPURLResponse {
            print("[HTTP RES][Swift] POST \(url.absoluteString) status=\(http.statusCode) body=\(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        return try jsonDecoder.decode(VoiceUploadResponse.self, from: data)
    }

    // MARK: - HTTP Streaming URLs
    func streamAudioURL(sessionId: String) -> URL? {
        return URL(string: "\(baseURL)/sessions/\(sessionId)/stream_audio")
    }
    
    func uploadAudioChunkURL(sessionId: String) -> URL? {
        return URL(string: "\(baseURL)/sessions/\(sessionId)/upload_audio")
    }
    
    func stopStreamURL(sessionId: String) -> URL? {
        return URL(string: "\(baseURL)/sessions/\(sessionId)/stop_stream")
    }
}
