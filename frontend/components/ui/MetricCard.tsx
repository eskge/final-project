
import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value }) => {
  return (
    <div className="bg-slate-700/50 p-4 rounded-lg text-center">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-2xl font-bold text-sky-400 mt-1">{value}</p>
    </div>
  );
};

export default MetricCard;
