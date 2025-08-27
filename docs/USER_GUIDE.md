# VerixAI User Guide

## Welcome to VerixAI

VerixAI is an intelligent document analysis assistant that helps you quickly extract insights, answers, and citations from large volumes of documents. Whether you're a doctor reviewing patient records, a lawyer researching case law, or an HR professional managing policies, VerixAI adapts to your needs.

## Getting Started

### First Time Setup

1. **Access the Application**
   - Open your web browser
   - Navigate to the VerixAI URL (e.g., http://localhost:3000)
   - You'll see the main dashboard with three tabs

2. **Understanding the Interface**
   - **Upload Documents**: Add new documents to the system
   - **Query Documents**: Ask questions about your documents
   - **Manage Datasets**: Organize and manage your document collections

## Uploading Documents

### Step-by-Step Upload Process

1. **Click the "Upload Documents" Tab**

2. **Select Your Files**
   - Drag and drop files into the upload area, OR
   - Click the upload area to browse and select files
   
   **Supported File Types:**
   - PDF documents
   - Microsoft Word (.docx)
   - PowerPoint presentations (.pptx)
   - HTML files
   - Text files (.txt)
   - Markdown files (.md)
   - Excel files (.xlsx)
   - JSON files (.json)

3. **Choose a Dataset**
   - **Create New Dataset**: Enter a descriptive name (e.g., "Q1-2024-reports")
   - **Add to Existing**: Select from your existing datasets

4. **Upload Your Files**
   - Click "Upload Documents"
   - Wait for processing (progress bar shows status)
   - You'll see a success message when complete

### Best Practices for Document Upload

- **Organize by Topic**: Group related documents in the same dataset
- **Use Clear Names**: "medical-records-2024" is better than "docs1"
- **Check File Quality**: Ensure PDFs are text-searchable (not scanned images)
- **Batch Upload**: Upload multiple related files at once for efficiency

## Querying Documents

### How to Ask Questions

1. **Navigate to "Query Documents" Tab**

2. **Select Your Context**
   - **Datasets**: Choose specific datasets to search (optional)
   - **Role**: Select your professional context:
     - **General**: Standard analysis
     - **Doctor**: Medical context with health disclaimers
     - **Lawyer**: Legal context with legal disclaimers
     - **HR**: HR/compliance focused responses

3. **Enter Your Question**
   - Type your question in natural language
   - Be specific for better results
   - Use the example queries as inspiration

4. **Review the Results**
   - **Answer**: AI-generated response with citations
   - **Confidence Level**: How certain the AI is about the answer
   - **Citations**: Source documents with relevance scores
   - **Key Highlights**: Bullet points of important information

### Example Queries by Role

#### General Queries
- "What are the main findings across all documents?"
- "What are the key dates and deadlines mentioned?"
- "List all recommendations provided"

#### Medical Professional Queries
- "What medications is the patient currently taking?"
- "What is the patient's medical history?"
- "Are there any noted allergies or contraindications?"
- "What were the results of the latest lab tests?"

#### Legal Professional Queries
- "Find all cases related to breach of contract"
- "What precedents support this argument?"
- "What are the key legal principles discussed?"
- "What damages were awarded in similar cases?"

#### HR Professional Queries
- "What are the current remote work policies?"
- "Compare vacation policies between 2023 and 2024"
- "List all mentioned compliance requirements"
- "What is the disciplinary process for policy violations?"

### Understanding Results

#### Answer Section
- Contains the AI's response to your question
- Includes inline citations like [Source 1]
- Formatted for easy reading

#### Citations
- Shows exact sources for information
- Includes relevance scores (higher is better)
- Click to expand and see the full context
- Shows which dataset and document chunk

#### Confidence Levels
- **High**: Strong evidence from multiple sources
- **Medium**: Good evidence but limited sources
- **Low**: Weak evidence or uncertain interpretation

## Managing Datasets

### Dataset Overview

Each dataset shows:
- **Name**: Your chosen identifier
- **Document Count**: Number of files
- **Size**: Total storage used
- **Created Date**: When first created

### Dataset Actions

1. **View Details**
   - Click the info icon for statistics
   - See file types and chunk counts

2. **Delete Dataset**
   - Click "Delete" button
   - Confirm in the dialog
   - **Warning**: This is permanent!

3. **Refresh List**
   - Click "Refresh" to update the view
   - Useful after uploads from another session

## Tips for Better Results

### Document Preparation
✅ **DO:**
- Use text-based PDFs (not scanned images)
- Ensure documents are complete
- Include relevant context documents
- Organize related documents together

❌ **DON'T:**
- Upload corrupted files
- Mix unrelated topics in one dataset
- Upload password-protected files
- Use special characters in dataset names

### Query Optimization
✅ **DO:**
- Ask specific, focused questions
- Mention key terms from your documents
- Use professional terminology when appropriate
- Try rephrasing if first results aren't ideal

❌ **DON'T:**
- Ask overly broad questions
- Expect analysis of images or charts
- Request information not in your documents
- Ignore the suggested follow-up questions

## Understanding Disclaimers

### Medical Disclaimer
When using the "Doctor" role, responses include:
> ⚠️ This information is for educational purposes only and is not a substitute for professional medical advice.

### Legal Disclaimer
When using the "Lawyer" role, responses include:
> ⚠️ This information is for educational purposes only and does not constitute legal advice.

### HR Disclaimer
When using the "HR" role, responses include:
> ⚠️ This information is for general guidance only. Consult with qualified professionals for specific situations.

## Privacy & Security

### Your Data
- Documents are processed locally in your deployment
- No data is sent to external services except for AI processing
- You control all document access and retention

### Best Practices
- Don't upload documents with passwords or sensitive credentials
- Regularly review and clean up old datasets
- Use appropriate role contexts for professional boundaries
- Verify critical information from original sources

## Troubleshooting

### Common Issues

**Upload Fails**
- Check file size (max 100MB per file)
- Verify file type is supported
- Ensure dataset name doesn't contain special characters
- Check your internet connection

**No Results Found**
- Verify documents were uploaded successfully
- Try broader search terms
- Check if you selected the correct dataset
- Ensure documents contain searchable text

**Slow Performance**
- Large documents take longer to process
- First queries may be slower (warming up)
- Check your internet connection
- Try uploading fewer files at once

**Incorrect Answers**
- Verify source documents contain the information
- Try rephrasing your question
- Check if you selected the appropriate role
- Review citations to understand the source

## Keyboard Shortcuts

- `Tab`: Navigate between sections
- `Enter`: Submit forms
- `Escape`: Close dialogs
- `Ctrl/Cmd + V`: Paste text

## Getting Help

### Support Resources
- Check this user guide first
- Review example queries for inspiration
- Contact your system administrator
- Report bugs through the feedback system

### Feedback
We welcome your feedback to improve VerixAI:
- Feature requests
- Bug reports
- Usability suggestions
- Documentation improvements

## Frequently Asked Questions

**Q: How many documents can I upload?**
A: There's no hard limit on document count, but individual files must be under 100MB.

**Q: Can I search across multiple datasets?**
A: Yes! In the Query tab, you can select multiple datasets to search simultaneously.

**Q: How accurate are the answers?**
A: Accuracy depends on document quality and question specificity. Always verify critical information.

**Q: Can I export results?**
A: You can copy answers and citations using the copy button for use in other applications.

**Q: Is my data secure?**
A: Yes, documents are stored securely and only accessible through the application.

**Q: Can I share datasets with colleagues?**
A: Currently, datasets are instance-wide. User-specific permissions are planned for future releases.

## Quick Reference Card

### File Types
- 📄 PDF, DOCX, PPTX
- 📝 TXT, MD, HTML
- 📊 XLSX, JSON

### Roles
- 👤 **General**: Standard analysis
- 👨‍⚕️ **Doctor**: Medical context
- ⚖️ **Lawyer**: Legal context
- 👥 **HR**: Compliance focus

### Confidence Indicators
- 🟢 **High**: Strong evidence
- 🟡 **Medium**: Moderate evidence
- 🔴 **Low**: Limited evidence