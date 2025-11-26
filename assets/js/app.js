let currentUtterance = null;
let speakingButton = null;

// Find and set the Kannada voice
let kannadaVoice = null;
function loadVoices() {
  const voices = window.speechSynthesis.getVoices();
  kannadaVoice = voices.find((voice) => voice.lang === "kn-IN");
}

// Load voices when they are ready
window.speechSynthesis.onvoiceschanged = loadVoices;
// Initial attempt to load voices
loadVoices();

function speakText(text, lang, buttonElement) {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    if (speakingButton && speakingButton !== buttonElement) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
  }

  if (speakingButton === buttonElement) {
    speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    speakingButton = null;
    currentUtterance = null;
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;

  // For Kannada, ensure the correct voice is used
  if (lang === "kn-IN" && kannadaVoice) {
    utterance.voice = kannadaVoice;
  }

  // Set the current button and icon
  speakingButton = buttonElement;
  speakingButton.innerHTML = '<i class="fas fa-stop"></i>';
  currentUtterance = utterance;

  utterance.onend = () => {
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    speakingButton = null;
    currentUtterance = null;
  };

  utterance.onerror = (event) => {
    console.error("Speech synthesis error:", event.error);
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    speakingButton = null;
    currentUtterance = null;
  };

  window.speechSynthesis.speak(utterance);
}

// Legacy function retained for compatibility, but new implementations should use speakText
function speak(text, lang, buttonElement) {
  speakText(text, lang, buttonElement);
}