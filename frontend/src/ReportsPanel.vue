<script setup>
import { ref } from 'vue'
import accuracyLatency from '../../docs/figures/accuracy-latency.png'
import boxRankBreakdown from '../../docs/figures/box-rank-breakdown.png'
import trainingRuns from '../../docs/figures/training-runs.png'

const panel = ref(null)
</script>

<template>
  <button type="button" class="quiet-button reports-trigger" aria-haspopup="dialog"
          aria-controls="reports-panel" @click="panel.showModal()">Reports</button>

  <dialog id="reports-panel" ref="panel" class="reports-panel" aria-labelledby="reports-title">
    <header class="reports-header">
      <h2 id="reports-title">Reports</h2>
      <button type="button" class="quiet-button" autofocus @click="panel.close()">Close</button>
    </header>

    <div class="reports-body">
      <section aria-labelledby="validation-title">
        <h3 id="validation-title">Uzbek visual grounding results</h3>
        <p>UzBEye compares inference-time translation with direct Uzbek grounding. All runs use
          the same 255 corrected Uzbek validation prompts for this comparison.</p>
        <figure>
          <img :src="accuracyLatency" width="1224" height="936"
               alt="Accuracy versus median latency. A thin line connects all four Model C runs. An inset separates three runs close to 205 milliseconds. Polished 8k C scores 56.5% at 236 milliseconds.">
          <figcaption>Top-1 accuracy counts a match when the first box overlaps the target with IoU ≥ 0.5.
            Latency was measured on a Quadro RTX 5000 and includes uncached translation for A.</figcaption>
        </figure>
      </section>

      <section aria-labelledby="ranking-title">
        <h3 id="ranking-title">Model C ranking experiment</h3>
        <p>The added loss encourages the best-overlapping box to score at least 0.10 higher than the strongest incorrect candidate.
          A matching box has IoU of at least 0.5.</p>
        <math class="ranking-equation" display="block"
              aria-label="Ranking loss equals the maximum of zero and 0.10 plus the wrong box score minus the best box score">
          <mrow>
            <msub><mi>L</mi><mtext>rank</mtext></msub>
            <mo>=</mo>
            <mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mn>0.10</mn><mo>+</mo>
            <msub><mi>s</mi><mtext>wrong</mtext></msub><mo>−</mo>
            <msub><mi>s</mi><mtext>best</mtext></msub><mo>)</mo>
          </mrow>
        </math>
        <p>On corrected 4k data, the ranking run raised top-1 from 41.6% to 48.2% and reduced top-5
          from 82.7% to 58.4%. The polished 8k follow-up reached 56.5% top-1 and 64.7% top-5
          with more training data and optimizer updates.</p>
        <figure>
          <img :src="boxRankBreakdown" width="1224" height="990" loading="lazy"
               alt="Polished 8k C ranked the matching box first on 144 prompts and in the top five on 165 prompts.">
          <figcaption>Each bar counts whether the first matching box was ranked first, ranked second to fifth,
            or absent from the top five.</figcaption>
        </figure>
      </section>

      <section aria-labelledby="training-title">
        <h3 id="training-title">Accuracy across checkpoints</h3>
        <figure>
          <img :src="trainingRuns" width="1224" height="1134" loading="lazy"
               alt="Top-1 accuracy and top-5 recall for Models B and C. Four Model C checkpoints are connected by a thin line in each plot.">
          <figcaption>Top-5 recall counts a match anywhere among the five highest-ranked boxes.</figcaption>
        </figure>
        <p class="report-note">The same validation set was used for checkpoint selection and scoring.</p>
      </section>
    </div>
  </dialog>
</template>

<style scoped>
:global(html:has(.reports-panel[open])) { overflow: hidden; }

.reports-trigger {
  padding: 7px 13px;
  font-weight: 630;
}

.reports-panel {
  --drawer-inset: clamp(8px, 2vw, 16px);
  inset: var(--drawer-inset) var(--drawer-inset) var(--drawer-inset) auto;
  margin: 0;
  padding: 0;
  width: min(44rem, calc(100% - 2 * var(--drawer-inset)));
  height: calc(100dvh - 2 * var(--drawer-inset));
  max-width: none;
  max-height: none;
  border: 1px solid var(--border);
  border-radius: var(--radius-panel);
  color: var(--ink);
  background: var(--panel);
  box-shadow: -18px 0 56px #0005;
  overscroll-behavior: contain;
}

.reports-panel::backdrop {
  background: #13130eb0;
}

.reports-header {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px clamp(16px, 4vw, 28px);
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}

.reports-header h2 {
  font-size: 1.2rem;
}

.reports-body {
  padding: 24px clamp(16px, 4vw, 28px) 32px;
  line-height: 1.55;
}

.reports-body section + section {
  margin-top: 26px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}

.reports-body h3 {
  margin: 0 0 8px;
  font-size: 1.05rem;
}

.reports-body p {
  margin: 0;
  color: var(--muted);
  font-size: .875rem;
}

.ranking-equation {
  margin: 10px 0 12px;
  color: var(--ink);
}

.reports-body figure {
  margin: 16px 0 0;
}

.reports-body img {
  display: block;
  width: 100%;
  height: auto;
  object-fit: contain;
  border: 1px solid var(--border);
  border-radius: var(--radius-control);
}

.reports-body figcaption {
  margin-top: 8px;
  color: var(--muted);
  font-size: .8125rem;
}

.reports-body .report-note {
  margin-top: 16px;
}
</style>
