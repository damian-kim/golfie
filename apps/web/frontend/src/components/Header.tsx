import { Link, useLocation } from "react-router-dom";
import "./Header.css";

const STEPS: { path: string; label: string }[] = [
  { path: "/", label: "Upload" },
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
      <svg className="app-header__tracer" viewBox="0 0 1000 12" preserveAspectRatio="none" aria-hidden="true">
        <line x1="0" y1="6" x2="1000" y2="6" stroke="var(--color-border)" strokeWidth="1" />
        <line
          x1="0"
          y1="6"
          x2="1000"
          y2="6"
          stroke="var(--color-turf-bright)"
          strokeWidth="1.5"
          strokeDasharray="2 10"
          opacity="0.55"
        />
        <circle cx="0" cy="6" r="3" fill="var(--color-turf-bright)" />
      </svg>
    </header>
  );
}
