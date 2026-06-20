import "./WarningsList.css";

interface WarningsListProps {
  warnings: string[];
}

export function WarningsList({ warnings }: WarningsListProps) {
  if (warnings.length === 0) return null;
  return (
    <div className="warnings-list">
      <div className="warnings-list__title">Warnings</div>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}
