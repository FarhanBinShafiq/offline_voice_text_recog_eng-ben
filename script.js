const questions = {
    'en-US': [
        {
            question: "What is the capital of France?",
            choices: ["Paris", "London", "Berlin", "Madrid"],
            answer: "Paris"
        },
        {
            question: "Which planet is known as the Red Planet?",
            choices: ["Mars", "Venus", "Jupiter", "Saturn"],
            answer: "Mars"
        },
        {
            question: "What is 2 + 2?",
            choices: ["3", "4", "5", "6"],
            answer: "4"
        }
    ],
    'bn-BD': [
        {
            question: "বাংলাদেশের রাজধানী কী?",
            choices: ["ঢাকা", "চট্টগ্রাম", "খুলনা", "রাজশাহী"],
            answer: "ঢাকা"
        },
        {
            question: "কোন নদীকে 'গঙ্গা' বলা হয়?",
            choices: ["পদ্মা", "মেঘনা", "যমুনা", "ব্রহ্মপুত্র"],
            answer: "পদ্মা"
        },
        {
            question: "২ + ২ এর সমষ্টি কত?",
            choices: ["৩", "৪", "৫", "৬"],
            answer: "৪"
        }
    ]
};

let currentQuestion = 0;
let currentLanguage = 'en-US';

const questionElement = document.getElementById('question');
const choicesElement = document.getElementById('choices');
const resultElement = document.getElementById('result');
const statusElement = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const languageSelect = document.getElementById('language-select');

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

function loadQuestion() {
    const q = questions[currentLanguage][currentQuestion];
    questionElement.textContent = q.question;
    choicesElement.innerHTML = '';
    q.choices.forEach(choice => {
        const div = document.createElement('div');
        div.className = 'choice';
        div.textContent = choice;
        div.onclick = () => checkAnswer(choice);
        choicesElement.appendChild(div);
    });
}

function checkAnswer(choice) {
    const q = questions[currentLanguage][currentQuestion];
    const correctMessage = currentLanguage === 'en-US' ? "Correct!" : "সঠিক!";
    const incorrectMessage = currentLanguage === 'en-US'
        ? `Incorrect. The correct answer is ${q.answer}.`
        : `ভুল। সঠিক উত্তর হল ${q.answer}।`;

    if (choice.toLowerCase() === q.answer.toLowerCase()) {
        resultElement.textContent = correctMessage;
        resultElement.style.color = "green";
    } else {
        resultElement.textContent = incorrectMessage;
        resultElement.style.color = "red";
    }
    setTimeout(() => {
        currentQuestion = (currentQuestion + 1) % questions[currentLanguage].length;
        loadQuestion();
        resultElement.textContent = '';
    }, 2000);
}

if (recognition) {
    recognition.continuous = false;
    recognition.interimResults = false;

    languageSelect.onchange = () => {
        currentLanguage = languageSelect.value;
        recognition.lang = currentLanguage;
        currentQuestion = 0;
        loadQuestion();
        statusElement.textContent = currentLanguage === 'en-US'
            ? "Click 'Start Voice Recognition' to begin or click an answer"
            : "শুরু করতে 'ভয়েস রেকগনিশন শুরু করুন' ক্লিক করুন অথবা একটি উত্তর ক্লিক করুন";
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.trim().toLowerCase();
        statusElement.textContent = currentLanguage === 'en-US'
            ? `You said: ${transcript}`
            : `আপনি বলেছেন: ${transcript}`;
        const q = questions[currentLanguage][currentQuestion];
        const matchedChoice = q.choices.find(choice =>
            choice.toLowerCase() === transcript ||
            transcript.includes(choice.toLowerCase())
        );
        if (matchedChoice) {
            checkAnswer(matchedChoice);
        } else {
            statusElement.textContent = currentLanguage === 'en-US'
                ? "Sorry, I didn't understand. Please try again or click an answer."
                : "দুঃখিত, আমি বুঝতে পারিনি। আবার চেষ্টা করুন বা একটি উত্তর ক্লিক করুন।";
        }
    };

    recognition.onend = () => {
        statusElement.textContent = currentLanguage === 'en-US'
            ? "Click 'Start Voice Recognition' to try again or click an answer"
            : "আবার চেষ্টা করতে 'ভয়েস রেকগনিশন শুরু করুন' ক্লিক করুন বা একটি উত্তর ক্লিক করুন";
        startBtn.textContent = currentLanguage === 'en-US'
            ? "Start Voice Recognition"
            : "ভয়েস রেকগনিশন শুরু করুন";
    };

    recognition.onerror = (event) => {
        statusElement.textContent = currentLanguage === 'en-US'
            ? "Speech recognition error. Try again or click an answer."
            : "ভয়েস রেকগনিশন ত্রুটি। আবার চেষ্টা করুন বা একটি উত্তর ক্লিক করুন।";
    };

    startBtn.onclick = () => {
        try {
            recognition.start();
            statusElement.textContent = currentLanguage === 'en-US'
                ? "Listening..."
                : "শুনছে...";
            startBtn.textContent = currentLanguage === 'en-US'
                ? "Listening..."
                : "শুনছে...";
        } catch (e) {
            statusElement.textContent = currentLanguage === 'en-US'
                ? "Speech recognition unavailable offline. Please click an answer."
                : "অফলাইনে ভয়েস রেকগনিশন অনুপলব্ধ। দয়া করে একটি উত্তর ক্লিক করুন।";
        }
    };
} else {
    startBtn.style.display = 'none';
    statusElement.textContent = currentLanguage === 'en-US'
        ? "Speech recognition not supported. Click choices to answer."
        : "ভয়েস রেকগনিশন সমর্থিত নয়। উত্তর দিতে পছন্দগুলি ক্লিক করুন।";
}

window.onload = () => {
    if (recognition) {
        recognition.lang = currentLanguage;
    }
    loadQuestion();
};