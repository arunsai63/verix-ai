import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface CardProps extends HTMLMotionProps<"div"> {
  variant?: 'default' | 'glass' | 'gradient' | 'outlined';
  hover?: boolean;
  glow?: boolean;
  children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({
  variant = 'default',
  hover = true,
  glow = false,
  children,
  className,
  ...props
}) => {
  const baseStyles = 'rounded-2xl transition-all duration-300';
  
  const variants = {
    default: 'bg-white dark:bg-neutral-900 shadow-lg border border-neutral-200 dark:border-neutral-700',
    glass: 'bg-white/80 dark:bg-neutral-900/80 backdrop-blur-sm border border-neutral-200 dark:border-neutral-700',
    gradient: 'bg-gradient-to-br from-white to-neutral-50 dark:from-neutral-900 dark:to-neutral-800 backdrop-blur-sm border border-neutral-200 dark:border-neutral-700 shadow-xl',
    outlined: 'bg-white/50 dark:bg-neutral-900/50 border-2 border-neutral-300 dark:border-neutral-700',
  };
  
  const hoverStyles = hover ? 'hover:shadow-2xl hover:scale-[1.02]' : '';
  const glowStyles = glow ? 'hover-glow' : '';
  
  const classes = twMerge(
    clsx(
      baseStyles,
      variants[variant],
      hoverStyles,
      glowStyles,
      className
    )
  );
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={classes}
      {...props}
    >
      {children}
    </motion.div>
  );
};

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ children, className }) => (
  <div className={twMerge('px-6 py-4 border-b border-neutral-200 dark:border-neutral-700', className)}>
    {children}
  </div>
);

interface CardBodyProps {
  children: React.ReactNode;
  className?: string;
}

export const CardBody: React.FC<CardBodyProps> = ({ children, className }) => (
  <div className={twMerge('p-6', className)}>
    {children}
  </div>
);

interface CardFooterProps {
  children: React.ReactNode;
  className?: string;
}

export const CardFooter: React.FC<CardFooterProps> = ({ children, className }) => (
  <div className={twMerge('px-6 py-4 border-t border-neutral-200 dark:border-neutral-700', className)}>
    {children}
  </div>
);

export const CardContent = CardBody;
export const CardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
  <h3 className={twMerge('text-lg font-semibold', className)}>{children}</h3>
);
export const CardDescription: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
  <p className={twMerge('text-sm text-gray-600 dark:text-gray-400', className)}>{children}</p>
);

export { Card };
export default Card;