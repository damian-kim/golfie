import { Routes, Route, Navigate } from "react-router-dom";
import { Header } from "./components/Header";
import { UploadPage } from "./pages/UploadPage";
import { ProcessingPage } from "./pages/ProcessingPage";
import { ReviewPage } from "./pages/ReviewPage";
import { SimulatorPage } from "./pages/SimulatorPage";
import { DemoPage } from "./pages/DemoPage";
import { CalibratePage } from "./pages/CalibratePage";
import { CoursePage } from "./pages/CoursePage";
import { AutoTrimPage } from "./pages/AutoTrimPage";

function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/calibrate" element={<CalibratePage />} />
        <Route path="/course" element={<CoursePage />} />
        <Route path="/video-editor" element={<AutoTrimPage />} />
        <Route path="/auto-trim" element={<Navigate to="/video-editor" replace />} />
        <Route path="/sessions/:sessionId/processing" element={<ProcessingPage />} />
        <Route path="/sessions/:sessionId/review" element={<ReviewPage />} />
        <Route path="/sessions/:sessionId/simulator" element={<SimulatorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
