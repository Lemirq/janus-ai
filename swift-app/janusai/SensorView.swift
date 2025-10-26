//
//  SensorView.swift
//  janusai
//
//  Created by Vihaan Sharma on 2025-10-26.
//

import SwiftUI
import Combine

#if canImport(CoreMotion)
import CoreMotion
#endif

class SensorManager: ObservableObject {
    #if canImport(CoreMotion)
    private let motionManager = CMMotionManager()
    #endif
    
    @Published var accelerometerX: Double = 0.0
    @Published var accelerometerY: Double = 0.0
    @Published var accelerometerZ: Double = 0.0
    
    @Published var gyroscopeX: Double = 0.0
    @Published var gyroscopeY: Double = 0.0
    @Published var gyroscopeZ: Double = 0.0
    
    @Published var isTracking = false
    @Published var accelerometerAvailable = false
    @Published var gyroscopeAvailable = false
    
    // Flick detection: detects quick wrist rotation using gyroscope Z axis
    @Published var flickDetected = false
    private var gyroscopeZValues: [Double] = []
    
    private var flickResetTimer: Timer?
    
    // Settings state
    @Published var currentSpeed: String = "medium"
    @Published var currentSpeedMultiplier: Double = 1.0
    @Published var verboseMode: Bool = false
    @Published var lastSettingsUpdate: String = ""
    
    // For gesture control
    private var previousVerboseZone: String? = nil  // "negative" or "positive"
    private var previousSpeedZone: String? = nil     // "slow", "medium", or "fast"
    
    // Debounce timers
    private var verboseDebounceTimer: Timer?
    private var speedDebounceTimer: Timer?
    
    // Session ID for repeat functionality (must be set externally)
    var currentSessionId: String? = nil
    
    init() {
        #if canImport(CoreMotion)
        accelerometerAvailable = motionManager.isAccelerometerAvailable
        gyroscopeAvailable = motionManager.isGyroAvailable
        #endif
    }
    
    private func detectFlick() {
        gyroscopeZValues.append(gyroscopeZ)
        
        // Keep only last 5 values (0.5 seconds at 0.1s intervals)
        if gyroscopeZValues.count > 5 {
            gyroscopeZValues.removeFirst()
        }
        
        // Need at least 5 samples to detect flick
        guard gyroscopeZValues.count == 5 else { return }
        
        // Calculate absolute of average of the 0.5s segment
        let average = abs(gyroscopeZValues.reduce(0, +) / Double(gyroscopeZValues.count))
        
        // Check if average exceeds threshold
        let threshold = 2.0
        if average > threshold {
            flickDetected = true
            scheduleFlickReset()
            handleFlickGesture()
        }
    }
    
    private func handleFlickGesture() {
        // Flick back gesture: replay last audio
        guard let sessionId = currentSessionId else {
            print("[SensorManager] Cannot replay: no active session")
            return
        }
        
        Task {
            do {
                let audioData = try await APIService.shared.repeatLastAudio(sessionId: sessionId)
                print("[SensorManager] Flick detected - replaying last audio (\(audioData.count) bytes)")
                // TODO: Play the audio data through AVAudioPlayer or similar
            } catch {
                print("[SensorManager] Failed to replay audio: \(error)")
            }
        }
    }
    
    private func handleAccelerometerGestures() {
        // Verbose mode control via Z axis
        // Z between -1 and 0 → verbose = false
        // Z between 0 and 1 → verbose = true
        let zZone = accelerometerZ < 0 ? "negative" : "positive"
        
        if previousVerboseZone != zZone {
            previousVerboseZone = zZone
            let newVerbose = (zZone == "positive")
            
            if newVerbose != verboseMode {
                // Cancel existing debounce timer
                verboseDebounceTimer?.invalidate()
                
                // Start new debounce timer (0.5 seconds)
                verboseDebounceTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { [weak self] _ in
                    guard let self = self else { return }
                    
                    Task {
                        do {
                            let settings = try await APIService.shared.updateSettings(verbose: newVerbose)
                            await MainActor.run {
                                self.updateSettingsState(settings)
                            }
                            print("[SensorManager] Z-axis gesture (debounced): verbose = \(newVerbose)")
                        } catch {
                            print("[SensorManager] Failed to update verbose: \(error)")
                        }
                    }
                }
            }
        }
        
        // Playback speed control via Y axis
        // slow: -1 < y < -0.33
        // med: -0.33 < y < 0.33
        // fast: 0.33 < y < 1
        let yZone: String
        if accelerometerY < -0.33 {
            yZone = "slow"
        } else if accelerometerY > 0.33 {
            yZone = "fast"
        } else {
            yZone = "medium"
        }
        
        if previousSpeedZone != yZone {
            previousSpeedZone = yZone
            
            if yZone != currentSpeed {
                // Cancel existing debounce timer
                speedDebounceTimer?.invalidate()
                
                // Start new debounce timer (0.5 seconds)
                speedDebounceTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { [weak self] _ in
                    guard let self = self else { return }
                    
                    Task {
                        do {
                            let settings = try await APIService.shared.updateSettings(playbackSpeed: yZone)
                            await MainActor.run {
                                self.updateSettingsState(settings)
                            }
                            print("[SensorManager] Y-axis gesture (debounced): speed = \(yZone)")
                        } catch {
                            print("[SensorManager] Failed to update speed: \(error)")
                        }
                    }
                }
            }
        }
    }
    
    private func updateSettingsState(_ settings: APIService.Settings) {
        currentSpeed = settings.playbackSpeed
        currentSpeedMultiplier = settings.speedMultiplier
        verboseMode = settings.verbose
        lastSettingsUpdate = settings.lastUpdated
    }
    
    func fetchSettings() {
        Task {
            do {
                let settings = try await APIService.shared.getSettings()
                await MainActor.run {
                    self.updateSettingsState(settings)
                }
            } catch {
                print("[SensorManager] Failed to fetch settings: \(error)")
            }
        }
    }
    
    private func scheduleFlickReset() {
        flickResetTimer?.invalidate()
        flickResetTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) { [weak self] _ in
            self?.flickDetected = false
        }
    }
    
    func startTracking() {
        guard !isTracking else { return }
        
        // Fetch initial settings
        fetchSettings()
        
        #if canImport(CoreMotion)
        // Start accelerometer updates
        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = 0.1
            motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
                guard let self = self, let data = data, error == nil else { return }
                self.accelerometerX = data.acceleration.x
                self.accelerometerY = data.acceleration.y
                self.accelerometerZ = data.acceleration.z
                self.handleAccelerometerGestures()
            }
        }
        
        // Start gyroscope updates
        if motionManager.isGyroAvailable {
            motionManager.gyroUpdateInterval = 0.1
            motionManager.startGyroUpdates(to: .main) { [weak self] data, error in
                guard let self = self, let data = data, error == nil else { return }
                self.gyroscopeX = data.rotationRate.x
                self.gyroscopeY = data.rotationRate.y
                self.gyroscopeZ = data.rotationRate.z
                self.detectFlick()
            }
        }
        #endif
        
        isTracking = true
    }
    
    func stopTracking() {
        #if canImport(CoreMotion)
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        #endif
        
        // Clean up all timers
        flickResetTimer?.invalidate()
        flickResetTimer = nil
        verboseDebounceTimer?.invalidate()
        verboseDebounceTimer = nil
        speedDebounceTimer?.invalidate()
        speedDebounceTimer = nil
        
        gyroscopeZValues.removeAll()
        flickDetected = false
        isTracking = false
    }
}

struct SensorView: View {
    @StateObject private var sensorManager = SensorManager()
    @EnvironmentObject private var appState: AppState
    
    var body: some View {
        NavigationView {
            ZStack(alignment: .top) {
                ScrollView {
                    VStack(spacing: 24) {
                        // Flick Detection Banner
                        if sensorManager.flickDetected {
                            FlickBanner()
                                .transition(.move(edge: .top).combined(with: .opacity))
                        }
                        
                        // Session Status
                        if let sessionId = appState.currentSessionId {
                            SessionStatusCard(
                                sessionId: sessionId,
                                isRunning: appState.isSessionRunning
                            )
                        }
                        
                        // Settings Display
                        SettingsCard(
                            speed: sensorManager.currentSpeed,
                            speedMultiplier: sensorManager.currentSpeedMultiplier,
                            verbose: sensorManager.verboseMode
                        )
                        
                        // Control Section
                        VStack(spacing: 16) {
                            HStack {
                                Image(systemName: sensorManager.isTracking ? "stop.circle.fill" : "play.circle.fill")
                                    .font(.title)
                                    .foregroundColor(sensorManager.isTracking ? .red : .green)
                                
                                Text(sensorManager.isTracking ? "Tracking Active" : "Tracking Stopped")
                                    .font(.headline)
                                
                                Spacer()
                            }
                            
                            Button(action: {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                    if sensorManager.isTracking {
                                        sensorManager.stopTracking()
                                    } else {
                                        sensorManager.startTracking()
                                    }
                                }
                            }) {
                                HStack {
                                    Image(systemName: sensorManager.isTracking ? "stop.fill" : "play.fill")
                                    Text(sensorManager.isTracking ? "Stop Tracking" : "Start Tracking")
                                        .fontWeight(.semibold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(sensorManager.isTracking ? Color.red : Color.green)
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            }
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                    
                        // Accelerometer Section
                        SensorCard(
                            title: "Accelerometer",
                            icon: "arrow.up.and.down.and.arrow.left.and.right",
                            available: sensorManager.accelerometerAvailable,
                            xValue: sensorManager.accelerometerX,
                            yValue: sensorManager.accelerometerY,
                            zValue: sensorManager.accelerometerZ,
                            unit: "g"
                        )
                        
                        // Gyroscope Section
                        SensorCard(
                            title: "Gyroscope",
                            icon: "gyroscope",
                            available: sensorManager.gyroscopeAvailable,
                            xValue: sensorManager.gyroscopeX,
                            yValue: sensorManager.gyroscopeY,
                            zValue: sensorManager.gyroscopeZ,
                            unit: "rad/s"
                        )
                        
                        // Info Section
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Sensor Information", systemImage: "info.circle")
                                .font(.headline)
                            
                            Text("The accelerometer measures acceleration forces including gravity. The gyroscope measures the rate of rotation around the device's axes.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                    }
                    .padding()
                }
                .background(Color(.systemGroupedBackground))
            }
            .navigationTitle("Sensors")
            .onDisappear {
                sensorManager.stopTracking()
            }
            .onAppear {
                // Sync current session ID on appear
                sensorManager.currentSessionId = appState.currentSessionId
            }
            .onChange(of: appState.currentSessionId) { newSessionId in
                // Update sensor manager when session changes
                sensorManager.currentSessionId = newSessionId
                print("[SensorView] Session ID updated: \(newSessionId ?? "nil")")
            }
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: sensorManager.flickDetected)
    }
}

struct FlickBanner: View {
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "hand.point.up.left.fill")
                .font(.title2)
                .foregroundColor(.white)
            
            Text("Flick Detected!")
                .font(.headline)
                .foregroundColor(.white)
            
            Spacer()
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.purple, Color.blue],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .purple.opacity(0.3), radius: 8, x: 0, y: 4)
    }
}

struct SensorCard: View {
    let title: String
    let icon: String
    let available: Bool
    let xValue: Double
    let yValue: Double
    let zValue: Double
    let unit: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(.accentColor)
                Text(title)
                    .font(.title3)
                    .fontWeight(.bold)
                Spacer()
                if !available {
                    Text("Not Available")
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.red.opacity(0.1))
                        .clipShape(Capsule())
                }
            }
            
            Divider()
            
            if available {
                VStack(spacing: 12) {
                    AxisView(label: "X", value: xValue, unit: unit, color: .red)
                    AxisView(label: "Y", value: yValue, unit: unit, color: .green)
                    AxisView(label: "Z", value: zValue, unit: unit, color: .blue)
                }
            } else {
                Text("This sensor is not available on this device.")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .padding(.vertical)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

struct AxisView: View {
    let label: String
    let value: Double
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text(label)
                    .font(.headline)
                    .foregroundColor(color)
                    .frame(width: 30, alignment: .leading)
                
                Text(String(format: "%.3f", value))
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.medium)
                    .frame(width: 100, alignment: .trailing)
                    .contentTransition(.numericText())
                
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 50, alignment: .leading)
                
                Spacer()
            }
            
            // Visual bar indicator
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 6)
                    
                    // Value indicator (normalized to -2 to 2 range)
                    let normalizedValue = max(-1, min(1, value / 2.0))
                    let centerOffset = geometry.size.width / 2
                    let barWidth = abs(normalizedValue) * centerOffset
                    let xOffset = normalizedValue > 0 ? centerOffset : centerOffset - barWidth
                    
                    RoundedRectangle(cornerRadius: 3)
                        .fill(
                            LinearGradient(
                                colors: [color, color.opacity(0.7)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: barWidth, height: 6)
                        .offset(x: xOffset)
                        .animation(.easeInOut(duration: 0.15), value: value)
                    
                    // Center marker
                    Rectangle()
                        .fill(Color.black.opacity(0.4))
                        .frame(width: 2, height: 10)
                        .offset(x: centerOffset - 1)
                }
            }
            .frame(height: 10)
        }
        .padding(.vertical, 4)
    }
}

struct SessionStatusCard: View {
    let sessionId: String
    let isRunning: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            // Status indicator
            Circle()
                .fill(isRunning ? Color.green : Color.orange)
                .frame(width: 12, height: 12)
                .shadow(color: isRunning ? .green : .orange, radius: 4)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("Active Session")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(sessionId)
                    .font(.system(.subheadline, design: .monospaced))
                    .fontWeight(.semibold)
            }
            
            Spacer()
            
            Text(isRunning ? "RUNNING" : "PAUSED")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(isRunning ? .green : .orange)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill((isRunning ? Color.green : Color.orange).opacity(0.2))
                )
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.blue.opacity(0.1), Color.cyan.opacity(0.05)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.blue.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .blue.opacity(0.15), radius: 6, x: 0, y: 3)
    }
}

struct SettingsCard: View {
    let speed: String
    let speedMultiplier: Double
    let verbose: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "gearshape.2.fill")
                    .font(.title2)
                    .foregroundColor(.orange)
                Text("Current Settings")
                    .font(.title3)
                    .fontWeight(.bold)
                Spacer()
            }
            
            Divider()
            
            HStack(spacing: 20) {
                // Speed indicator
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: speedIcon)
                            .foregroundColor(speedColor)
                        Text("Speed")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Text(speed.capitalized)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(speedColor)
                    
                    Text("\(String(format: "%.2f", speedMultiplier))x")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                
                Divider()
                    .frame(height: 60)
                
                // Verbose indicator
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "text.bubble.fill")
                            .foregroundColor(verbose ? .green : .gray)
                        Text("Verbose")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Text(verbose ? "ON" : "OFF")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(verbose ? .green : .gray)
                    
                    Text(verbose ? "Detailed" : "Quiet")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            // Gesture instructions
            VStack(alignment: .leading, spacing: 6) {
                Text("Gesture Controls")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
                
                HStack(spacing: 4) {
                    Image(systemName: "hand.point.up.left.fill")
                        .font(.caption2)
                    Text("Flick: Replay audio")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
                
                HStack(spacing: 4) {
                    Image(systemName: "arrow.up.and.down")
                        .font(.caption2)
                    Text("Y-axis: Speed control")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
                
                HStack(spacing: 4) {
                    Image(systemName: "arrow.left.and.right")
                        .font(.caption2)
                    Text("Z-axis: Verbose mode")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
            }
            .padding(.top, 8)
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.orange.opacity(0.1), Color.yellow.opacity(0.05)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .orange.opacity(0.15), radius: 8, x: 0, y: 4)
    }
    
    private var speedIcon: String {
        switch speed {
        case "slow": return "tortoise.fill"
        case "fast": return "hare.fill"
        default: return "gauge.medium"
        }
    }
    
    private var speedColor: Color {
        switch speed {
        case "slow": return .blue
        case "fast": return .red
        default: return .green
        }
    }
}

#Preview {
    SensorView()
}
