
import React from 'react';
import { Prediction } from '../../types';
import Button from './Button';

interface ResultTableProps {
  predictions: Prediction[];
}

const ResultTable: React.FC<ResultTableProps> = ({ predictions }) => {
  const downloadCSV = () => {
    const header = ['customerID', 'Predicted_Churn', 'Churn_Probability'];
    const rows = predictions.map(p => [p.customerID, p.Predicted_Churn, p.Churn_Probability]);
    const csvContent = "data:text/csv;charset=utf-8," 
      + [header.join(','), ...rows.map(e => e.join(','))].join('\n');
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'churn_predictions.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div>
        <div className="flex justify-end mb-4">
            <Button onClick={downloadCSV}>
                Download CSV
            </Button>
        </div>
        <div className="overflow-x-auto relative shadow-md sm:rounded-lg">
            <table className="w-full text-sm text-left text-slate-300">
                <thead className="text-xs text-slate-200 uppercase bg-slate-700/50">
                    <tr>
                        <th scope="col" className="py-3 px-6">Customer ID</th>
                        <th scope="col" className="py-3 px-6">Predicted Churn</th>
                        <th scope="col" className="py-3 px-6">Churn Probability</th>
                    </tr>
                </thead>
                <tbody>
                    {predictions.map((p, index) => (
                        <tr key={index} className="border-b border-slate-700 hover:bg-slate-700/30">
                            <th scope="row" className="py-4 px-6 font-medium text-white whitespace-nowrap">
                                {p.customerID}
                            </th>
                            <td className="py-4 px-6">
                                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                                    p.Predicted_Churn === 'Yes' 
                                    ? 'bg-red-500/20 text-red-300' 
                                    : 'bg-green-500/20 text-green-300'
                                }`}>
                                    {p.Predicted_Churn}
                                </span>
                            </td>
                            <td className="py-4 px-6 font-mono">
                                {p.Churn_Probability}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
  );
};

export default ResultTable;
