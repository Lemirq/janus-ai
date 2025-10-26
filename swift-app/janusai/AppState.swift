//
//  AppState.swift
//  janusai
//
//  Created by Assistant on 2025-10-26.
//

import Foundation
import Combine

/// Shared app state for tracking current session and settings
final class AppState: ObservableObject {
    /// The currently active session ID, or nil if no session is running
    @Published var currentSessionId: String? = nil
    
    /// Whether a session is currently running
    @Published var isSessionRunning: Bool = false
    
    /// Singleton instance
    static let shared = AppState()
    
    private init() {}
    
    /// Call when a session starts
    func startSession(_ sessionId: String) {
        print("[AppState] Session started: \(sessionId)")
        self.currentSessionId = sessionId
        self.isSessionRunning = true
    }
    
    /// Call when a session stops
    func stopSession() {
        print("[AppState] Session stopped: \(currentSessionId ?? "none")")
        self.isSessionRunning = false
    }
    
    /// Call when a session completes (clears the session ID)
    func completeSession() {
        print("[AppState] Session completed: \(currentSessionId ?? "none")")
        self.currentSessionId = nil
        self.isSessionRunning = false
    }
}

