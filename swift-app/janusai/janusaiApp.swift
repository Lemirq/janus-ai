//
//  janusaiApp.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-24.
//

import SwiftUI

@main
struct janusaiApp: App {
    @StateObject private var appState = AppState.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}
