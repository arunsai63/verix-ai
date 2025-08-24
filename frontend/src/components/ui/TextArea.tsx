import React from 'react';

interface TextAreaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

const TextArea: React.FC<TextAreaProps> = ({
  label,
  error,
  helperText,
  className = '',
  ...props
}) => {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1.5">
          {label}
        </label>
      )}
      <textarea
        className={`
          w-full px-4 py-3 rounded-xl border transition-all duration-200
          bg-white dark:bg-neutral-800
          text-neutral-900 dark:text-neutral-100 font-medium
          placeholder:text-neutral-400 dark:placeholder:text-neutral-500
          ${error 
            ? 'border-error-500 focus:ring-error-500' 
            : 'border-neutral-300 dark:border-neutral-700 focus:ring-primary-500 focus:border-primary-500'
          }
          focus:outline-none focus:ring-2 focus:border-transparent
          disabled:opacity-50 disabled:cursor-not-allowed
          resize-vertical
          ${className}
        `}
        {...props}
      />
      {(error || helperText) && (
        <p className={`mt-1.5 text-sm ${error ? 'text-error-600 dark:text-error-400' : 'text-neutral-600 dark:text-neutral-400'}`}>
          {error || helperText}
        </p>
      )}
    </div>
  );
};

export default TextArea;