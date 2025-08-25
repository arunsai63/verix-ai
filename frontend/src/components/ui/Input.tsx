import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  variant?: 'default' | 'filled' | 'outlined';
  inputSize?: 'sm' | 'md' | 'lg';
}

const Input: React.FC<InputProps> = ({
  label,
  error,
  leftIcon,
  rightIcon,
  variant = 'default',
  inputSize = 'md',
  className,
  ...props
}) => {
  const baseStyles = 'w-full transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent';
  
  const variants = {
    default: 'bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700 text-neutral-900 dark:text-neutral-100 font-medium placeholder:text-neutral-400 dark:placeholder:text-neutral-500',
    filled: 'bg-neutral-100 dark:bg-neutral-900 border border-transparent text-neutral-900 dark:text-neutral-100 font-medium placeholder:text-neutral-400 dark:placeholder:text-neutral-500',
    outlined: 'bg-transparent dark:bg-neutral-800/50 border-2 border-neutral-300 dark:border-neutral-700 text-neutral-900 dark:text-neutral-100 font-medium placeholder:text-neutral-400 dark:placeholder:text-neutral-500',
  };
  
  const sizes = {
    sm: 'px-3 py-2 text-sm rounded-lg',
    md: 'px-4 py-3 text-base rounded-xl',
    lg: 'px-5 py-4 text-lg rounded-xl',
  };
  
  const errorStyles = error ? 'border-error-500 focus:ring-error-500' : '';
  
  const inputClasses = twMerge(
    clsx(
      baseStyles,
      variants[variant],
      sizes[inputSize],
      errorStyles,
      leftIcon && 'pl-10',
      rightIcon && 'pr-10',
      className
    )
  );
  
  return (
    <div className="w-full">
      {label && (
        <motion.label
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2"
        >
          {label}
        </motion.label>
      )}
      <div className="relative">
        {leftIcon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400">
            {leftIcon}
          </div>
        )}
        <input
          className={inputClasses}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neutral-400">
            {rightIcon}
          </div>
        )}
      </div>
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 text-sm text-error-600"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
};

export { Input };
export default Input;