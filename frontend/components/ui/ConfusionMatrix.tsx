import React from 'react';
import { ConfusionMatrixData } from '../../types';

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ data }) => {
  // FIX: Corrected destructuring syntax for a nested array.
  const [[tn, fp], [fn, tp]] = data;

  return (
    <div className="flex flex-col items-center">
      <div className="flex items-center">
        <div className="w-24 text-center transform -rotate-90 text-sm font-semibold text-slate-300">Actual</div>
        <div className="flex flex-col">
          <div className="grid grid-cols-3 gap-1">
            <div className="p-2"></div>
            <div className="text-center font-semibold text-slate-300 col-span-2">Predicted</div>
            <div className="text-center text-sm font-semibold text-slate-300 py-2">No Churn</div>
            <div className="bg-green-500/10 text-green-300 p-4 rounded text-center text-2xl font-mono">{tn}</div>
            <div className="bg-red-500/10 text-red-300 p-4 rounded text-center text-2xl font-mono">{fp}</div>
          </div>
          <div className="grid grid-cols-3 gap-1 mt-1">
             <div className="text-center text-sm font-semibold text-slate-300 py-2">Churn</div>
             <div className="bg-red-500/10 text-red-300 p-4 rounded text-center text-2xl font-mono">{fn}</div>
             <div className="bg-green-500/10 text-green-300 p-4 rounded text-center text-2xl font-mono">{tp}</div>
          </div>
        </div>
      </div>
       <div className="mt-4 text-xs text-slate-400 text-center">
            <p><span className="text-green-400 font-semibold">Green:</span> Correct Predictions (True Positives & True Negatives)</p>
            <p><span className="text-red-400 font-semibold">Red:</span> Incorrect Predictions (False Positives & False Negatives)</p>
        </div>
    </div>
  );
};

export default ConfusionMatrix;
