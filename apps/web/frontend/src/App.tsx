import { Routes, Route, Navigate } from "react-router-dom";
import { Header } from "./components/Header";
import { UploadPage } from "./pages/UploadPage";
import { ProcessingPage } from "./pages/ProcessingPage";
import { ReviewPage } from "./pages/ReviewPage";
import { SimulatorPage } from "./pages/SimulatorPage";
import { DemoPage } from "./pages/DemoPage";

function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/sessions/:sessionId/processing" element={<ProcessingPage />} />
        <Route path="/sessions/:sessionId/review" element={<ReviewPage />} />
        <Route path="/sessions/:sessionId/simulator" element={<SimulatorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
