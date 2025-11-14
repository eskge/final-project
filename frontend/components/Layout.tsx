
import React from 'react';
import { Page } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  page: Page;
  setPage: (page: Page) => void;
}

interface NavItemProps {
  label: Page;
  icon: React.ReactNode;
  currentPage: Page;
  setPage: (page: Page) => void;
}

const HomeIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
);

const PipelineIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12h2"/><path d="M7 12h2"/><path d="M12 12h2"/><path d="M17 12h2"/><path d="M22 12h-2"/><path d="M12 2v2"/><path d="M12 7v2"/><path d="M12 17v2"/><path d="M12 22v-2"/><path d="m15 9-3 3-3-3"/><path d="m15 15-3-3-3 3"/></svg>
);

const PredictIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"></path><path d="m9 12 2 2 4-4"></path></svg>
);

const ReportIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line></svg>
);


const NavItem: React.FC<NavItemProps> = ({ label, icon, currentPage, setPage }) => {
  const isActive = currentPage === label;
  return (
    <button
      onClick={() => setPage(label)}
      className={`flex items-center w-full px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-200 ${
        isActive
          ? 'bg-sky-500 text-white'
          : 'text-slate-400 hover:bg-slate-700 hover:text-slate-100'
      }`}
    >
      <span className="mr-3">{icon}</span>
      {label}
    </button>
  );
};

const Layout: React.FC<LayoutProps> = ({ children, page, setPage }) => {
  return (
    <div className="flex h-screen bg-slate-800">
      <aside className="w-64 flex-shrink-0 bg-slate-900 p-4 border-r border-slate-700 flex flex-col">
        <div className="flex items-center mb-8">
          <div className="bg-sky-500 p-2 rounded-lg mr-3">
             <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          </div>
          <h1 className="text-xl font-bold text-white">Churn Predictor</h1>
        </div>
        <nav className="flex flex-col space-y-2">
          <NavItem label="Home" icon={<HomeIcon />} currentPage={page} setPage={setPage} />
          <NavItem label="Pipeline" icon={<PipelineIcon />} currentPage={page} setPage={setPage} />
          <NavItem label="Prediction" icon={<PredictIcon />} currentPage={page} setPage={setPage} />
          <NavItem label="Reports" icon={<ReportIcon />} currentPage={page} setPage={setPage} />
        </nav>
         <div className="mt-auto text-center text-xs text-slate-500">
            <p>E2E Data Science Project</p>
            <p>&copy; 2024</p>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        {children}
      </main>
    </div>
  );
};

export default Layout;
