import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface SkeletonProps {
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  className?: string;
  animation?: 'pulse' | 'shimmer' | 'none';
}

const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'text',
  width,
  height,
  className,
  animation = 'shimmer',
}) => {
  const baseStyles = 'bg-neutral-200';
  
  const variants = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-none',
    rounded: 'rounded-lg',
  };
  
  const animations = {
    pulse: 'animate-pulse',
    shimmer: 'skeleton',
    none: '',
  };
  
  const defaultSizes = {
    text: { width: '100%', height: '1em' },
    circular: { width: '40px', height: '40px' },
    rectangular: { width: '100%', height: '120px' },
    rounded: { width: '100%', height: '120px' },
  };
  
  const finalWidth = width || defaultSizes[variant].width;
  const finalHeight = height || defaultSizes[variant].height;
  
  const classes = twMerge(
    clsx(
      baseStyles,
      variants[variant],
      animations[animation],
      className
    )
  );
  
  return (
    <div
      className={classes}
      style={{
        width: typeof finalWidth === 'number' ? `${finalWidth}px` : finalWidth,
        height: typeof finalHeight === 'number' ? `${finalHeight}px` : finalHeight,
      }}
    />
  );
};

interface SkeletonGroupProps {
  count?: number;
  children: React.ReactNode;
  className?: string;
}

export const SkeletonGroup: React.FC<SkeletonGroupProps> = ({
  count = 1,
  children,
  className,
}) => {
  return (
    <div className={twMerge('space-y-3', className)}>
      {Array.from({ length: count }).map((_, index) => (
        <div key={index}>{children}</div>
      ))}
    </div>
  );
};

export default Skeleton;