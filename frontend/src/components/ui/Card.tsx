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
    default: 'bg-white shadow-lg border border-neutral-100',
    glass: 'glass',
    gradient: 'bg-gradient-to-br from-white/90 to-white/70 backdrop-blur-sm border border-white/50 shadow-xl',
    outlined: 'bg-transparent border-2 border-neutral-200',
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
  <div className={twMerge('px-6 py-4 border-b border-neutral-100', className)}>
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
  <div className={twMerge('px-6 py-4 border-t border-neutral-100', className)}>
    {children}
  </div>
);

export default Card;