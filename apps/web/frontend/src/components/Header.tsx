import { Link, useLocation } from "react-router-dom";
import "./Header.css";

const STEPS: { path: string; label: string }[] = [
  { path: "/", label: "Upload" },
  { path: "/calibrate", label: "Calibrate" },
  { path: "/demo", label: "Demo range" },
];

export function Header() {
  const location = useLocation();
  return (
    <header className="app-header">
      <div className="app-header__row">
        <Link to="/" className="app-header__wordmark">
          GOLFIE
        </Link>
        <nav className="app-header__nav">
          {STEPS.map((step) => (
            <Link
              key={step.path}
              to={step.path}
              className={
                location.pathname === step.path
                  ? "app-header__link app-header__link--active"
                  : "app-header__link"
              }
            >
              {step.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
