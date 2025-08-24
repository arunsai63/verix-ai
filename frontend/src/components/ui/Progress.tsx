import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ProgressProps {
  value: number;
  max?: number;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'gradient';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  className?: string;
}

const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  variant = 'primary',
  size = 'md',
  showLabel = false,
  label,
  animated = true,
  className,
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  const baseStyles = 'relative overflow-hidden bg-neutral-200 rounded-full';
  
  const sizes = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4',
  };
  
  const barVariants = {
    default: 'bg-neutral-500',
    primary: 'bg-gradient-to-r from-primary-500 to-primary-600',
    success: 'bg-gradient-to-r from-success-500 to-success-600',
    warning: 'bg-gradient-to-r from-warning-500 to-warning-600',
    error: 'bg-gradient-to-r from-error-500 to-error-600',
    gradient: 'bg-gradient-to-r from-primary-500 via-accent-500 to-success-500',
  };
  
  const containerClasses = twMerge(
    clsx(
      baseStyles,
      sizes[size],
      className
    )
  );
  
  return (
    <div className="w-full">
      {(showLabel || label) && (
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium text-neutral-700">
            {label || 'Progress'}
          </span>
          <span className="text-sm text-neutral-500">
            {percentage.toFixed(0)}%
          </span>
        </div>
      )}
      <div className={containerClasses}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{
            duration: animated ? 0.5 : 0,
            ease: 'easeOut',
          }}
          className={clsx(
            'h-full rounded-full relative overflow-hidden',
            barVariants[variant]
          )}
        >
          {animated && (
            <div className="absolute inset-0 bg-white/30 animate-shimmer" />
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default Progress;