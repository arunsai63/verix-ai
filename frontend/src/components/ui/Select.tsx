import React from 'react';
import { ChevronDown } from 'lucide-react';

interface SelectOption {
  value: string;
  label: string;
  description?: string;
}

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  label?: string;
  options: SelectOption[];
  value: string | string[];
  onChange: (value: string | string[]) => void;
  multiple?: boolean;
  error?: string;
  helperText?: string;
}

const Select: React.FC<SelectProps> = ({
  label,
  options,
  value,
  onChange,
  multiple = false,
  error,
  helperText,
  disabled,
  className = '',
  ...props
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (multiple) {
      const selectedOptions = Array.from(e.target.selectedOptions, option => option.value);
      onChange(selectedOptions);
    } else {
      onChange(e.target.value);
    }
  };

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          value={value}
          onChange={handleChange}
          multiple={multiple}
          disabled={disabled}
          className={`
            w-full px-4 py-3 pr-10 rounded-xl border transition-all duration-200
            bg-white dark:bg-neutral-800
            text-neutral-900 dark:text-neutral-100
            ${error 
              ? 'border-error-500 dark:border-error-400 focus:ring-error-500' 
              : 'border-neutral-300 dark:border-neutral-700 focus:ring-primary-500'
            }
            focus:outline-none focus:ring-2 focus:border-transparent
            disabled:opacity-50 disabled:cursor-not-allowed
            ${multiple ? 'min-h-[120px]' : ''}
            appearance-none
            ${className}
          `}
          {...props}
        >
          {!multiple && (
            <option value="" disabled>
              Select an option...
            </option>
          )}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {!multiple && (
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400 pointer-events-none" />
        )}
      </div>
      {(error || helperText) && (
        <p className={`mt-1.5 text-sm ${error ? 'text-error-600 dark:text-error-400' : 'text-neutral-600 dark:text-neutral-400'}`}>
          {error || helperText}
        </p>
      )}
    </div>
  );
};

export default Select;