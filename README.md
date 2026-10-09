# Green AI: Lifecycle Assessment of ML Models

An interactive, zero-build web application designed to evaluate the entire lifecycle carbon footprint and energy consumption of machine learning models for production deployment and AutoML architectures.

Aligned with **Green AI** principles (Schwartz et al., 2020), **ISO/IEC 21031:2024**, and the **Green Software Foundation (GSF) SCI for AI** specification.

> 🚀 **Live Demo**: [Click here to see a live demo](https://dsachar.net/GreenAI)

---

## 🌟 Overview

Traditional AutoML and model selection heuristics rank candidate models based solely on offline validation accuracy over static benchmarks ("Red AI"). Once deployed in production:
- Real-world data drift degrades brittle architectures, requiring costly retraining cycles.
- Each retraining event repeatedly incurs the model's full training compute, energy, and carbon footprint.
- Inference traffic scales continuously over millions of operational predictions.
- **Software Carbon Intensity for AI (SCI for AI)** captures operational electricity ($O$, serving + retraining) and embodied hardware manufacturing footprint ($M$).
- **Consumer SCI (QA)** accounts for model accuracy under drift, measuring carbon per 1,000 *correct* predictions delivered.

---

## 🚀 Getting Started

The project is a zero-build static web application (HTML5, CSS3, ES6+ JavaScript) with no build tools or package managers required.

To run locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/dsachar/GreenAI.git
   cd GreenAI
   ```
2. Serve the directory using any static file server (or open `index.html` directly in a browser):
   ```bash
   python3 -m http.server 8080
   ```
3. Open `http://localhost:8080` in your browser.

---

## 📁 Project Structure

- `index.html`: Workflow UI with interactive tabs (*01. Models*, *02. Configuration*, *03. Assessment*, *04. Which Model to Deploy*, and *Methodology*).
- `styles.css`: Custom "field-notebook" responsive design system.
- `app.js`: State management, $O(1)$ memory closed-form simulation engine, GSF SCI accounting, and bounded SVG chart renderers.
- `info.md`: Detailed documentation and mathematical formulation.

---

## 📖 Documentation

For in-depth mathematical formulations, notation tables, and simulation model details, see [info.md](info.md).

---

## 📄 License

MIT
