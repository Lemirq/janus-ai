//
//  WaveformView.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI

struct WaveformView: View {
    var samples: [Float] // normalized 0...1

    var color: Color = .accentColor

    var body: some View {
        GeometryReader { geo in
            let barWidth = max(1, geo.size.width / CGFloat(max(samples.count, 1)))
            HStack(spacing: 1) {
                ForEach(Array(samples.enumerated()), id: \.offset) { _, v in
                    let h = CGFloat(max(0, min(1, v))) * geo.size.height
                    Rectangle()
                        .fill(color)
                        .frame(width: barWidth, height: max(2, h))
                        .cornerRadius(1)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
        .clipped()
    }
}


