# Gesture Debouncing Implementation

## Overview

Added 0.5-second debouncing to speed and verbose gesture controls to prevent API spam from frequent accelerometer updates.

## Problem

- Accelerometer updates every 0.1 seconds
- Users may hover near threshold values, causing rapid zone transitions
- Without debouncing, this would trigger multiple API calls per second
- Could overwhelm the backend and waste battery/bandwidth

## Solution

### Debounce Timers

Added two timer properties to `SensorManager`:

```swift
private var verboseDebounceTimer: Timer?
private var speedDebounceTimer: Timer?
```

### How It Works

#### Before (No Debouncing)

```
Accelerometer update → Zone change detected → Immediate API call
   (0.1s later)     → Zone change detected → Immediate API call
   (0.1s later)     → Zone change detected → Immediate API call
```

#### After (With 0.5s Debouncing)

```
Accelerometer update → Zone change detected → Start 0.5s timer
   (0.1s later)     → Zone change detected → Cancel timer, start new 0.5s timer
   (0.1s later)     → Zone change detected → Cancel timer, start new 0.5s timer
   (0.3s passes)    → No more changes       → Timer fires → API call ✓
```

### Implementation Details

#### Verbose Mode Debouncing

```swift
if previousVerboseZone != zZone {
    previousVerboseZone = zZone
    let newVerbose = (zZone == "positive")

    if newVerbose != verboseMode {
        // Cancel existing timer
        verboseDebounceTimer?.invalidate()

        // Start new 0.5s timer
        verboseDebounceTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { [weak self] _ in
            // API call happens here after 0.5s of stability
            Task { ... }
        }
    }
}
```

#### Speed Mode Debouncing

Same pattern as verbose mode, but for Y-axis speed zones.

### Timer Cleanup

All timers are properly cleaned up when tracking stops:

```swift
func stopTracking() {
    flickResetTimer?.invalidate()
    verboseDebounceTimer?.invalidate()
    speedDebounceTimer?.invalidate()
    // ... reset other state
}
```

## Benefits

1. **Reduces API calls**: Only sends request after user settles in a zone
2. **Better UX**: Prevents flashing UI from rapid state changes
3. **Battery efficient**: Fewer network requests
4. **Backend friendly**: Prevents request spam
5. **More intentional**: Only triggers when user holds position for 0.5s

## User Experience

### Scenario: User tilts device slowly

```
Y = -0.35 (slow zone)   → Timer starts
Y = -0.32 (medium zone) → Timer canceled, new timer starts
Y = -0.30 (medium zone) → Timer continues
... 0.5s passes with Y in medium zone ...
→ API call: Set speed to "medium" ✓
```

### Scenario: User overshoots then corrects

```
Y = 0.5 (fast zone)     → Timer starts
Y = 0.3 (medium zone)   → Timer canceled, new timer starts
Y = 0.2 (medium zone)   → Timer continues
... 0.5s passes with Y in medium zone ...
→ API call: Set speed to "medium" ✓
```

Result: Only one API call, for the intended final position.

## Testing

To verify debouncing works:

1. Enable verbose mode (`verbose: true` in settings)
2. Start sensor tracking
3. Tilt device slowly through zones
4. Check console logs - should see:
   - Zone changes detected frequently
   - API call only happens after 0.5s of stability
   - Log message includes "(debounced)"

Example log output:

```
[SensorManager] Y-axis gesture (debounced): speed = slow
[SensorManager] Z-axis gesture (debounced): verbose = true
```

## Configuration

Current debounce delay: **0.5 seconds**

To adjust, change `withTimeInterval` parameter:

```swift
Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false)
                                      // ↑ Change this value
```

Recommended range: 0.3-1.0 seconds

- Too short: Still too many API calls
- Too long: Feels unresponsive
