<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const approaches = [
  { id: 'A', name: 'Translate-Test', description: 'Translates the prompt to English.' },
  { id: 'B', name: 'Translate-Train', description: 'Fine-tuned on Uzbek prompts.' },
  { id: 'C', name: 'TahrirchiBERT', description: 'Uses a fine-tuned Uzbek text encoder.' },
  { id: 'D', name: 'Zero-Shot Uzbek', description: 'Runs the pretrained model directly on Uzbek.' },
]
const samples = [
  { name: 'Giraffes', file: '/samples/giraffes.jpg', prompt: "Yuzi ko'rinib turgan jirafa." },
  { name: 'Horse', file: '/samples/horse.jpg', prompt: "Otga qo'lini tekkizib turgan ayol." },
  { name: 'Sandwich', file: '/samples/sandwich.jpg', prompt: 'Tepadagi sendvich.' },
]
const selection = ref('C')
const sample = ref(samples[0])
const prompt = ref(samples[0].prompt)
const uploaded = ref(null)
const objectUrl = ref('')
const filePicker = ref(null)
const device = ref('')
const availability = ref({})
const connectionError = ref(false)
const busy = ref(false)
const dragging = ref(false)
const result = ref(null)
const selectedRank = ref(0)
const error = ref('')
const shape = ref(null)

const imageUrl = computed(() => objectUrl.value || sample.value.file)
const filename = computed(() => uploaded.value?.name || sample.value.name)
const activePrediction = computed(() => result.value?.predictions[selectedRank.value])
const frameStyle = computed(() => ({ '--ratio': shape.value ? shape.value.width / shape.value.height : 4 / 3 }))
const selectedModel = computed(() => approaches.find(model => model.id === selection.value))
const ready = computed(() => shape.value && availability.value[selection.value] && prompt.value.trim())
const score = computed(() => activePrediction.value ? (activePrediction.value.score * 100).toFixed(1) + '%' : '')
const boxStyle = computed(() => {
  if (!activePrediction.value) return {}
  const [x1, y1, x2, y2] = activePrediction.value.box
  return { left: x1 * 100 + '%', top: y1 * 100 + '%', width: (x2 - x1) * 100 + '%', height: (y2 - y1) * 100 + '%' }
})

function clearResult() {
  result.value = null
  selectedRank.value = 0
  error.value = ''
}

function releaseUpload() {
  if (objectUrl.value) URL.revokeObjectURL(objectUrl.value)
  objectUrl.value = ''
  uploaded.value = null
}

function chooseSample(item) {
  if (uploaded.value || sample.value.file !== item.file) shape.value = null
  releaseUpload()
  sample.value = item
  prompt.value = item.prompt
  clearResult()
}

function chooseFile(event) {
  const file = event.target.files[0]
  event.target.value = ''
  useFile(file)
}

function dropFile(event) {
  dragging.value = false
  useFile(event.dataTransfer.files[0])
}

function useFile(file) {
  if (!file || busy.value) return
  clearResult()
  if (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    error.value = 'Choose a JPG, PNG or WebP image.'
    return
  }
  if (file.size > 12 * 1024 * 1024) {
    error.value = 'Choose an image under 12 MB.'
    return
  }
  releaseUpload()
  shape.value = null
  uploaded.value = file
  objectUrl.value = URL.createObjectURL(file)
}

function imageLoaded(event) {
  shape.value = { width: event.target.naturalWidth, height: event.target.naturalHeight }
}

function imageFailed() {
  shape.value = null
  error.value = 'This image could not be opened.'
}

async function connect() {
  connectionError.value = false
  try {
    const response = await fetch('/api/models')
    if (!response.ok) throw new Error()
    const data = await response.json()
    device.value = data.device
    availability.value = data.models
    if (!availability.value[selection.value]) selection.value = 'D'
  } catch {
    connectionError.value = true
  }
}

async function ground() {
  if (!ready.value || busy.value) return
  clearResult()
  busy.value = true
  try {
    const body = new FormData()
    body.set('approach', selection.value)
    body.set('prompt', prompt.value.trim())
    let picture = uploaded.value
    if (!picture) {
      const sampleResponse = await fetch(sample.value.file)
      if (!sampleResponse.ok) throw new Error('Could not load this sample.')
      picture = await sampleResponse.blob()
    }
    body.set('image', picture, uploaded.value?.name || sample.value.name + '.jpg')
    const response = await fetch('/api/predict', { method: 'POST', body })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the image and prompt, then try again.')
    }
    result.value = data
  } catch (err) {
    error.value = err instanceof SyntaxError ? 'The local server could not complete inference. Check its terminal.' : err.message
  } finally {
    busy.value = false
  }
}

onMounted(connect)
onUnmounted(releaseUpload)
</script>

<template>
  <header class="topbar">
    <div class="shell topbar-content">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <h1>UzBEye</h1>
      </div>
      <div class="device" aria-live="polite">
        <span class="device-indicator" :class="{ online: device }" aria-hidden="true"></span>
        {{ device || (connectionError ? 'Server offline' : 'Connecting') }}
        <button v-if="connectionError" type="button" class="retry" @click="connect">Retry</button>
      </div>
    </div>
  </header>

  <main class="shell">
    <div class="intro">
      <p><strong>Exploring Accuracy–Efficiency Trade-offs in Uzbek Visual Grounding</strong></p>
    </div>

    <div class="workspace">
      <form class="panel controls" @submit.prevent="ground">
        <fieldset :disabled="busy">
          <legend>Input</legend>
          <div class="image-label">
            <label for="image">Image</label>
            <button type="button" class="quiet-button" @click="filePicker.click()">Upload image</button>
            <input id="image" ref="filePicker" class="visually-hidden" tabindex="-1" type="file"
                   accept="image/jpeg,image/png,image/webp" @change="chooseFile">
          </div>
          <div class="samples" aria-label="Sample images">
            <button v-for="item in samples" :key="item.file" type="button" class="sample"
                    :aria-pressed="!uploaded && sample.file === item.file" @click="chooseSample(item)">
              <img :src="item.file" alt="" decoding="async">
              <span>{{ item.name }}</span>
            </button>
          </div>

          <label class="prompt-label" for="prompt">Uzbek prompt</label>
          <textarea id="prompt" v-model="prompt" rows="2" maxlength="500"
                    placeholder="Masalan, chapdagi oq soyabon" @input="clearResult"></textarea>

          <span id="models-label" class="model-label">Model</span>
          <div class="models" role="radiogroup" aria-labelledby="models-label" aria-describedby="model-description">
            <label v-for="model in approaches" :key="model.id" class="model" :class="{ active: selection === model.id }">
              <input v-model="selection" type="radio" name="approach" :value="model.id"
                     :disabled="availability[model.id] === false" @change="clearResult">
              <span class="model-id">{{ model.id }}</span>
              <span class="model-name">{{ model.name }}</span>
              <span v-if="availability[model.id] === false" class="visually-hidden">Checkpoint needed</span>
            </label>
          </div>
          <p id="model-description" class="model-description">{{ selectedModel.description }}</p>
        </fieldset>
        <button class="primary" type="submit" :disabled="!ready || busy">
          <span v-if="busy" class="spinner" aria-hidden="true"></span>
          {{ busy ? 'Grounding…' : 'Run grounding' }}
        </button>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </form>

      <section class="panel output" aria-labelledby="output-title" :aria-busy="busy">
        <header class="output-header">
          <div>
            <h2 id="output-title">Prediction</h2>
            <p>{{ filename }}<span v-if="shape" class="dimensions"> · {{ shape.width }} × {{ shape.height }}</span></p>
          </div>
          <span class="status" :class="{ 'has-result': result }">
            {{ result ? selectedModel.name : busy ? 'Running' : 'Preview' }}
          </span>
        </header>
        <div class="stage" :class="{ 'is-dragging': dragging }"
             @dragover.prevent="dragging = !busy" @dragleave="dragging = false" @drop.prevent="dropFile">
          <div class="frame" :style="frameStyle">
            <img :key="imageUrl" :src="imageUrl" :alt="filename" draggable="false"
                 @load="imageLoaded" @error="imageFailed">
            <div v-if="activePrediction" class="box" :style="boxStyle" aria-hidden="true">
              <span>{{ score }}</span>
            </div>
          </div>
          <span v-if="dragging" class="drop-message">Drop an image to upload</span>
        </div>

        <div v-if="result" class="output-footer" aria-live="polite">
          <div class="result-metadata">
            <span><strong>{{ score }}</strong> box score</span>
            <span><strong>{{ Math.round(result.duration_ms) }} ms</strong> inference</span>
          </div>
          <div class="ranks" aria-label="Ranked boxes">
            <span>Box</span>
            <button v-for="(_, index) in result.predictions" :key="index" type="button"
                    :aria-label="'Show box ranked ' + (index + 1)" :aria-pressed="selectedRank === index"
                    @click="selectedRank = index">{{ index + 1 }}</button>
          </div>
          <p v-if="result.approach === 'A'" class="translation"><strong>English translation</strong> {{ result.grounded_prompt }}</p>
        </div>
        <div v-else class="output-footer empty" aria-live="polite">
          <p>{{ busy ? 'The first run can take longer while the model loads.' : 'Run grounding to see the five highest-ranked boxes.' }}</p>
        </div>
      </section>
    </div>
  </main>
</template>
