
# In silico Trial - Streamlit Application

![Banner](image1.jpg)

## Table of Contents
1. Introduction
2. Usage
3. Page Navigation
4. Landing Page
5. Initialization
6. Usability Test
7. AI Trial
8. AI + Explanation Trial
9. Final Pages
10. End Questionnaire
11. Video Instructions

---

### Introduction

This Streamlit application is designed for conducting an in silico trial related to post-hepatectomy liver failure prediction. The trial involves different phases, including usability testing, clinical trials with AI models, and AI models with explanations. The application allows researchers and participants to navigate through these phases seamlessly.

### Usage

To use this Streamlit application, follow these steps:

1. Ensure you have the required dependencies and datasets.
2. Run the application using Streamlit.

```bash
streamlit run Main_page.py
```

3. Follow the on-screen instructions to participate in the in silico trial.

### Page Navigation

The application uses a page navigation system to guide users through different phases of the in silico trial. The following pages are available:

- Landing Page
- Initialization
- Usability Test
- AI Trial
- AI + Explanation Trial
- Final Pages
- End Questionnaire

### Landing Page

The landing page serves as the entry point to the in silico trial. It displays the trial title, description, and buttons to start different types of trials:

- Usability Test
- Clinical Trial: AI Only
- Clinical Trial: AI + Explanation

The page also provides information about the timer, which starts when a trial is initiated.

### Initialization

This page initializes the trial by recording the start time and setting the session's page number. It is not directly accessible but is used internally to start trials.

### Usability Test

The usability test phase is designed to evaluate the usability of explanations. Participants can interact with the application to perform specific tasks and provide feedback on their experience.

### AI Trial

In the AI trial phase, participants are presented with AI-generated predictions. This phase evaluates the performance of the AI model.

### AI + Explanation Trial

In the AI + Explanation trial phase, participants receive AI-generated predictions along with explanations for the predictions. This phase assesses the effectiveness of explanations in aiding decision-making.

### Final Pages

The final pages provide summaries and conclusions for each phase of the trial. Participants can review their performance and experiences.

### End Questionnaire

Participants are required to complete an end questionnaire to provide feedback on their overall experience in the trial.

### Video Instructions
The application includes video instructions that participants can watch before beginning the trial. The video provides additional information and guidance.

The live version of this application is available at: [https://st-trial-hbhdzvtdqlr.streamlit.app/](https://st-trial-hbhdzvtdqlr.streamlit.app/)

