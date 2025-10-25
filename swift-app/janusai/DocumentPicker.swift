//
//  DocumentPicker.swift
//  janusai
//
//  Created by Assistant on 2025-10-25.
//

import SwiftUI
import UniformTypeIdentifiers
#if canImport(PDFKit)
import PDFKit
#endif

struct PickedDocument: Identifiable {
    let id = UUID()
    let url: URL
    let text: String
}

struct DocumentPicker: UIViewControllerRepresentable {
    var contentTypes: [UTType] = [.plainText, .utf8PlainText, .utf16PlainText, .text, .rtf, .pdf]
    var allowsMultipleSelection: Bool = true
    var onPick: ([PickedDocument]) -> Void
    var onCancel: () -> Void = {}

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let controller = UIDocumentPickerViewController(forOpeningContentTypes: contentTypes)
        controller.allowsMultipleSelection = allowsMultipleSelection
        controller.delegate = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick, onCancel: onCancel)
    }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: ([PickedDocument]) -> Void
        let onCancel: () -> Void

        init(onPick: @escaping ([PickedDocument]) -> Void, onCancel: @escaping () -> Void) {
            self.onPick = onPick
            self.onCancel = onCancel
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            onCancel()
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            var picked: [PickedDocument] = []
            for url in urls {
                guard url.startAccessingSecurityScopedResource() else { continue }
                defer { url.stopAccessingSecurityScopedResource() }
                if let text = Self.extractText(from: url) {
                    picked.append(PickedDocument(url: url, text: text))
                }
            }
            onPick(picked)
        }

        private static func extractText(from url: URL) -> String? {
            let type = UTType(filenameExtension: url.pathExtension) ?? .data
            if type.conforms(to: .plainText) || type == .rtf {
                return try? String(contentsOf: url)
            }
#if canImport(PDFKit)
            if type == .pdf, let pdf = PDFDocument(url: url) {
                return pdf.string
            }
#endif
            return nil
        }
    }
}


