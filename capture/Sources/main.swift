import AVFoundation
import Foundation
@preconcurrency import ScreenCaptureKit

// MARK: - Configuration

struct Config {
    var output = "recording.wav"
    var duration: Double? = nil
    var chunkSeconds: Double = 0
    var overlapSeconds: Double = 5
}

func parseArgs() -> Config {
    var c = Config()
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--output", "-o":
            i += 1; if i < args.count { c.output = args[i] }
        case "--duration", "-d":
            i += 1; if i < args.count { c.duration = Double(args[i]) }
        case "--chunk-seconds":
            i += 1; if i < args.count { c.chunkSeconds = Double(args[i]) ?? 0 }
        case "--overlap-seconds":
            i += 1; if i < args.count { c.overlapSeconds = Double(args[i]) ?? 5 }
        case "--help", "-h":
            fputs("""
                Usage: appaudio-capture [OPTIONS]
                  --output, -o PATH         Output WAV path (default: recording.wav)
                  --duration, -d SEC        Recording duration (omit for manual stop)
                  --chunk-seconds SEC       Chunk duration for pipelined output (0=disabled)
                  --overlap-seconds SEC     Overlap between chunks (default: 5)

                """, stderr)
            exit(0)
        default: break
        }
        i += 1
    }
    return c
}

// MARK: - Audio Recorder

class AudioRecorder: NSObject, SCStreamOutput {
    private var audioFile: AVAudioFile?
    private let outputPath: String
    private let chunkSeconds: Double
    private let overlapSeconds: Double

    private var chunkIndex = 0
    private var framesInChunk: Int64 = 0
    private var totalFrames: Int64 = 0
    private var rate: Double = 48000
    private var channels: UInt32 = 2
    private var fmt: AVAudioFormat?
    private var started = false

    // Overlap ring buffer
    private struct Buf { let data: Data; let frames: UInt32 }
    private var overlapRing: [Buf] = []
    private var overlapFrames: Int64 = 0

    var isChunking: Bool { chunkSeconds > 0 }
    private var framesPerChunk: Int64 { Int64(chunkSeconds * rate) }
    private var maxOverlapFrames: Int64 { Int64(overlapSeconds * rate) }

    init(outputPath: String, chunkSeconds: Double = 0, overlapSeconds: Double = 5) {
        self.outputPath = outputPath
        self.chunkSeconds = chunkSeconds
        self.overlapSeconds = overlapSeconds
        super.init()
    }

    private func chunkPath(_ idx: Int) -> String {
        guard isChunking else { return outputPath }
        let url = URL(fileURLWithPath: outputPath)
        let base = url.deletingPathExtension().path
        let ext = url.pathExtension.isEmpty ? "wav" : url.pathExtension
        return String(format: "%@_%03d.%@", base, idx, ext)
    }

    private func openFile(at path: String) {
        guard let f = fmt else { return }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: rate,
            AVNumberOfChannelsKey: Int(channels),
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            audioFile = try AVAudioFile(
                forWriting: URL(fileURLWithPath: path),
                settings: settings,
                commonFormat: f.commonFormat,
                interleaved: false
            )
        } catch {
            fputs("Error creating \(path): \(error)\n", stderr)
        }
    }

    private func writePCM(_ data: Data, frames: UInt32) {
        guard let f = fmt,
              let pcm = AVAudioPCMBuffer(pcmFormat: f, frameCapacity: frames) else { return }
        pcm.frameLength = frames
        data.withUnsafeBytes { buf in
            memcpy(pcm.mutableAudioBufferList.pointee.mBuffers.mData!, buf.baseAddress!, data.count)
        }
        try? audioFile?.write(from: pcm)
    }

    private func emitChunk() {
        let path = chunkPath(chunkIndex)
        audioFile = nil
        print(path)
        fflush(stdout)
        fputs("  Chunk \(chunkIndex) ready\n", stderr)
    }

    private func rotateChunk() {
        emitChunk()
        chunkIndex += 1
        framesInChunk = 0
        openFile(at: chunkPath(chunkIndex))
        for buf in overlapRing {
            writePCM(buf.data, frames: buf.frames)
        }
    }

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, sampleBuffer.isValid,
              let desc = sampleBuffer.formatDescription,
              let asbd = desc.audioStreamBasicDescription else { return }

        if !started {
            rate = asbd.mSampleRate
            channels = asbd.mChannelsPerFrame
            fmt = AVAudioFormat(
                commonFormat: asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
                    ? .pcmFormatFloat32 : .pcmFormatInt16,
                sampleRate: rate,
                channels: AVAudioChannelCount(channels),
                interleaved: false
            )
            started = true
            fputs("Audio: \(Int(rate)) Hz, \(channels) ch\n", stderr)
            openFile(at: chunkPath(chunkIndex))
        }

        guard let block = sampleBuffer.dataBuffer else { return }
        let len = block.dataLength
        let bpf = Int(asbd.mBytesPerFrame)
        guard bpf > 0 else { return }
        let frames = UInt32(len / bpf)

        var data = Data(count: len)
        data.withUnsafeMutableBytes { buf in
            CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: len, destination: buf.baseAddress!)
        }

        writePCM(data, frames: frames)
        framesInChunk += Int64(frames)
        totalFrames += Int64(frames)

        if isChunking {
            overlapRing.append(Buf(data: data, frames: frames))
            overlapFrames += Int64(frames)
            while overlapFrames > maxOverlapFrames, !overlapRing.isEmpty {
                overlapFrames -= Int64(overlapRing.removeFirst().frames)
            }
            if framesInChunk >= framesPerChunk {
                rotateChunk()
            }
        }
    }

    func close() {
        if isChunking {
            emitChunk()
        } else {
            audioFile = nil
        }
        fputs("Total: \(String(format: "%.1f", Double(totalFrames) / rate))s\n", stderr)
    }
}

// MARK: - Main

let config = parseArgs()
let recorder = AudioRecorder(
    outputPath: config.output,
    chunkSeconds: config.chunkSeconds,
    overlapSeconds: config.overlapSeconds
)

let semaphore = DispatchSemaphore(value: 0)
var stopped = false
let stopLock = NSLock()
var programExitCode: Int32 = 0
var activeStream: SCStream?

func stopOnce(cancel: Bool = false) {
    stopLock.lock()
    guard !stopped else { stopLock.unlock(); return }
    stopped = true
    stopLock.unlock()
    if cancel { programExitCode = 1 }
    Task {
        if let s = activeStream { try? await s.stopCapture() }
        recorder.close()
        semaphore.signal()
    }
}

// SIGINT handler — set up before any async work
signal(SIGINT, SIG_IGN)
let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
sigSource.setEventHandler {
    fputs("\nCancelled.\n", stderr)
    stopOnce(cancel: true)
}
sigSource.resume()

// Enter key handler
Thread.detachNewThread {
    while true {
        var buf = [UInt8](repeating: 0, count: 1)
        let n = read(STDIN_FILENO, &buf, 1)
        if n <= 0 || buf[0] == 0x0A {
            stopOnce()
            return
        }
    }
}

Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            fputs("Error: No display found\n", stderr)
            exit(1)
        }

        let sc = SCStreamConfiguration()
        sc.capturesAudio = true
        sc.excludesCurrentProcessAudio = false
        sc.sampleRate = 48000
        sc.channelCount = 2
        sc.width = 2
        sc.height = 2
        sc.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let stream = SCStream(filter: filter, configuration: sc, delegate: nil)
        let audioQueue = DispatchQueue(label: "audio-handler", qos: .userInitiated)
        try stream.addStreamOutput(recorder, type: .audio, sampleHandlerQueue: audioQueue)

        activeStream = stream
        try await stream.startCapture()

        fputs("Recording to \(config.output)... ", stderr)
        if let dur = config.duration {
            fputs("(\(Int(dur))s)\n", stderr)
            DispatchQueue.global().asyncAfter(deadline: .now() + dur) { stopOnce() }
        } else {
            fputs("(Press Enter to stop, Ctrl+C to cancel)\n", stderr)
        }
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

semaphore.wait()
exit(programExitCode)
