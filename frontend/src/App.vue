<template>
  <div class="app-container">
    <!-- Left Panel - Challenges -->
    <div class="panel challenges-panel">
      <h2 class="panel-title">Challenges</h2>
      <ul class="challenge-list">
        <li 
          v-for="challenge in challenges" 
          :key="challenge"
          class="challenge-item"
          :class="{ active: selectedChallenge === challenge }"
          @click="selectChallenge(challenge)"
        >
          {{ challenge }}
        </li>
      </ul>
    </div>

    <!-- Center Panel - Puzzle Viewer -->
    <div class="panel puzzle-panel">
      <h2 class="panel-title">Puzzle Viewer</h2>
      
      <div v-if="currentPuzzle">
        <h3>{{ currentPuzzle.id }}</h3>
        
        <!-- Training Examples -->
        <div v-for="(example, index) in currentPuzzle.train" :key="'train-' + index">
          <h4>Training Example {{ index + 1 }}</h4>
          <div class="grid-container">
            <div class="grid">
              <div class="grid-row" v-for="(row, rowIndex) in example.input" :key="rowIndex">
                <div 
                  class="grid-cell" 
                  v-for="(cell, colIndex) in row" 
                  :key="colIndex"
                  :style="{ backgroundColor: getColor(cell), color: getTextColor(cell) }"
                >
                  {{ cell }}
                </div>
              </div>
            </div>
            
            <div class="arrow">→</div>
            
            <div class="grid" v-if="example.output">
              <div class="grid-row" v-for="(row, rowIndex) in example.output" :key="rowIndex">
                <div 
                  class="grid-cell" 
                  v-for="(cell, colIndex) in row" 
                  :key="colIndex"
                  :style="{ backgroundColor: getColor(cell), color: getTextColor(cell) }"
                >
                  {{ cell }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Test Cases -->
        <div v-for="(test, index) in currentPuzzle.test" :key="'test-' + index">
          <h4>Test Case {{ index + 1 }}</h4>
          <div class="grid-container">
            <div class="grid">
              <div class="grid-row" v-for="(row, rowIndex) in test.input" :key="rowIndex">
                <div 
                  class="grid-cell" 
                  v-for="(cell, colIndex) in row" 
                  :key="colIndex"
                  :style="{ backgroundColor: getColor(cell), color: getTextColor(cell) }"
                >
                  {{ cell }}
                </div>
              </div>
            </div>
            <div class="arrow">→ ?</div>
          </div>
        </div>

        <button class="analyze-button" @click="analyzePuzzle">Analyze Pattern</button>
        
        <div class="info-section" v-if="analysis">
          <h4>Analysis</h4>
          <p><strong>Training Examples:</strong> {{ analysis.training_examples }}</p>
          <p><strong>Test Cases:</strong> {{ analysis.test_cases }}</p>
          <p><strong>Grid Size:</strong> {{ analysis.grid_size }}</p>
          <p><strong>Colors Used:</strong> {{ analysis.colors_used?.join(', ') }}</p>
          <p><strong>Patterns:</strong> {{ analysis.patterns?.join(', ') }}</p>
        </div>
      </div>
      
      <div v-else>
        <p>Select a challenge to view the puzzle</p>
      </div>
    </div>

    <!-- Right Panel - Chat -->
    <div class="panel chat-panel">
      <h2 class="panel-title">AI Assistant</h2>
      
      <div class="chat-messages" ref="chatMessages">
        <div 
          v-for="(message, index) in chatHistory" 
          :key="index"
          class="message"
          :class="message.type"
        >
          <div class="message-header">{{ message.timestamp }} - {{ message.sender }}</div>
          <div>{{ message.content }}</div>
        </div>
      </div>
      
      <div class="chat-input-container">
        <textarea 
          v-model="chatInput"
          class="chat-input"
          placeholder="Ask about the puzzle patterns..."
          @keydown="handleKeyDown"
          :disabled="isLoading"
        ></textarea>
        <button 
          class="send-button" 
          @click="sendMessage"
          :disabled="!chatInput.trim() || isLoading"
        >
          {{ isLoading ? 'Sending...' : 'Send' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'App',
  data() {
    return {
      challenges: [],
      selectedChallenge: null,
      currentPuzzle: null,
      analysis: null,
      chatHistory: [
        {
          type: 'system',
          sender: 'System',
          content: 'AI Assistant ready! Select a puzzle and ask questions.',
          timestamp: new Date().toLocaleTimeString()
        }
      ],
      chatInput: '',
      isLoading: false,
      colors: {
        0: '#000000',  // Black
        1: '#0074D9',  // Blue
        2: '#FF4136',  // Red
        3: '#2ECC40',  // Green
        4: '#FFDC00',  // Yellow
        5: '#AAAAAA',  // Gray
        6: '#F012BE',  // Fuchsia
        7: '#FF851B',  // Orange
        8: '#7FDBFF',  // Aqua
        9: '#870C25',  // Maroon
      }
    }
  },
  async mounted() {
    await this.loadChallenges()
  },
  methods: {
    async loadChallenges() {
      try {
        const response = await axios.get('/api/challenges')
        this.challenges = response.data.challenges
      } catch (error) {
        console.error('Error loading challenges:', error)
      }
    },
    
    async selectChallenge(challengeId) {
      this.selectedChallenge = challengeId
      try {
        const response = await axios.get(`/api/challenges/${challengeId}`)
        this.currentPuzzle = response.data
        this.analysis = null
        
        this.addMessage('system', `Loaded puzzle: ${challengeId}`)
      } catch (error) {
        console.error('Error loading challenge:', error)
      }
    },
    
    async analyzePuzzle() {
      if (!this.selectedChallenge) return
      
      try {
        const response = await axios.post(`/api/analyze/${this.selectedChallenge}`)
        this.analysis = response.data
        
        this.addMessage('system', 'Pattern analysis completed!')
      } catch (error) {
        console.error('Error analyzing puzzle:', error)
      }
    },
    
    handleKeyDown(event) {
      // Send message on Enter (but not Shift+Enter)
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        this.sendMessage()
      }
    },
    
    async sendMessage() {
      if (!this.chatInput.trim() || this.isLoading) return
      
      const message = this.chatInput.trim()
      this.chatInput = ''
      
      // Add user message
      this.addMessage('user', message)
      
      // Send to API
      this.isLoading = true
      try {
        const response = await axios.post('/api/chat', {
          message: message,
          challenge_id: this.selectedChallenge
        })
        
        this.addMessage('assistant', response.data.response)
      } catch (error) {
        console.error('Error sending message:', error)
        this.addMessage('system', 'Error: Could not send message to AI')
      } finally {
        this.isLoading = false
      }
    },
    
    addMessage(type, content) {
      this.chatHistory.push({
        type: type,
        sender: type === 'user' ? 'You' : type === 'assistant' ? 'AI' : 'System',
        content: content,
        timestamp: new Date().toLocaleTimeString()
      })
      
      // Scroll to bottom
      this.$nextTick(() => {
        const chatMessages = this.$refs.chatMessages
        chatMessages.scrollTop = chatMessages.scrollHeight
      })
    },
    
    getColor(value) {
      return this.colors[value] || '#FFFFFF'
    },
    
    getTextColor(value) {
      // Use white text for dark colors
      return [0, 1, 9].includes(value) ? '#FFFFFF' : '#000000'
    }
  }
}
</script>
