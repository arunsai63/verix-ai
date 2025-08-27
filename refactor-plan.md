# Major Refactor Plan - Focus on Core Document Processing

## Goal
Remove chat with document, summarization, and CSV analytics features to focus solely on the core product:
- Process large numbers of documents
- Handle queries with advanced ranking
- Provide effective results with citations

## Todo List

### 1. Analyze codebase to identify all chat and summarization features ✅
- [x] Identify backend chat/summarization endpoints
- [x] Identify frontend chat/summarization components
- [x] Identify related tests
- [x] Identify documentation references

### 2. Remove chat/summarization backend code and endpoints ✅
- [ ] Remove chat-related API endpoints
- [ ] Remove summarization API endpoints
- [ ] Remove related service methods
- [ ] Clean up agent code if needed

### 3. Remove chat/summarization frontend components and pages ✅
- [ ] Remove chat components
- [ ] Remove summarization components
- [ ] Remove related pages/routes
- [ ] Update navigation

### 4. Update API routes and remove unused endpoints ✅
- [ ] Clean up route definitions
- [ ] Remove unused imports
- [ ] Update API documentation

### 5. Clean up backend tests related to removed features ✅
- [ ] Remove chat-related tests
- [ ] Remove summarization tests
- [ ] Update integration tests

### 6. Clean up frontend tests related to removed features ✅
- [ ] Remove component tests
- [ ] Remove page tests
- [ ] Update integration tests

### 7. Update landing page to focus on core query functionality ✅
- [ ] Update hero section
- [ ] Update feature descriptions
- [ ] Update demo section
- [ ] Remove references to chat/summarization

### 8. Update documentation in docs folder ✅
- [ ] Update feature documentation
- [ ] Update API documentation
- [ ] Update user guides
- [ ] Update configuration guides

### 9. Update README.md to reflect product focus ✅
- [ ] Update project description
- [ ] Update features list
- [ ] Update usage instructions
- [ ] Update architecture description

### 10. Remove unused dependencies and imports ✅
- [ ] Clean backend dependencies
- [ ] Clean frontend dependencies
- [ ] Remove unused imports

### 11. Test the refactored application ✅
- [ ] Test document upload
- [ ] Test query functionality
- [ ] Test citation generation
- [ ] Run all remaining tests

## Files to Modify/Delete

### Backend
- API routes for chat/summarization
- Service methods for chat/summarization
- Tests for removed features
- Agent code related to summarization (if not used for queries)

### Frontend
- Chat components
- Summarization components
- Related pages and routes
- Tests for removed features

### Documentation
- docs/ folder content
- README.md
- Landing page content

## Core Features to Preserve
✓ Document upload and processing
✓ Query handling with advanced ranking
✓ Multi-agent retrieval system
✓ Citation generation
✓ Dataset management
✓ Parallel document processing