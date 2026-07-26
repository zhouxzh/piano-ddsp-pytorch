class PcmQueueProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.queue = [];
    this.offset = 0;
    this.queuedSamples = 0;
    this.primed = false;
    this.baseStartThreshold = Math.max(
      128,
      Number(options.processorOptions?.startThreshold || 2048),
    );
    this.maxStartThreshold = Math.max(
      this.baseStartThreshold,
      Number(options.processorOptions?.maxStartThreshold || this.baseStartThreshold * 3),
    );
    this.startThreshold = this.baseStartThreshold;
    this.reportCountdown = 0;
    this.port.onmessage = (event) => {
      if (event.data.type === "samples") {
        const samples = event.data.samples;
        this.queue.push(samples);
        this.queuedSamples += samples.length;
      } else if (event.data.type === "clear") {
        this.queue = [];
        this.offset = 0;
        this.queuedSamples = 0;
        this.primed = false;
        this.startThreshold = this.baseStartThreshold;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.primed && this.queuedSamples >= this.startThreshold) {
      this.primed = true;
    }
    if (this.primed) {
      let target = 0;
      while (target < output.length && this.queue.length) {
        const source = this.queue[0];
        const available = source.length - this.offset;
        const count = Math.min(available, output.length - target);
        output.set(source.subarray(this.offset, this.offset + count), target);
        target += count;
        this.offset += count;
        this.queuedSamples -= count;
        if (this.offset === source.length) {
          this.queue.shift();
          this.offset = 0;
        }
      }
      if (target < output.length) {
        this.primed = false;
        this.startThreshold = Math.min(
          this.maxStartThreshold,
          Math.ceil(this.startThreshold * 1.5),
        );
        this.port.postMessage({
          type: "underrun",
          restartThresholdMilliseconds: this.startThreshold * 1000 / sampleRate,
        });
      }
    }
    this.reportCountdown -= output.length;
    if (this.reportCountdown <= 0) {
      this.reportCountdown = sampleRate / 4;
      this.port.postMessage({
        type: "buffer",
        milliseconds: this.queuedSamples * 1000 / sampleRate,
      });
    }
    return true;
  }
}

registerProcessor("pcm-queue", PcmQueueProcessor);
