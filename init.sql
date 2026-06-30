-- =============================================================================
-- init.sql — PostgreSQL initialization
-- Runs automatically on first container startup
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable full-text search utilities
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create database (if not using default)
-- CREATE DATABASE IF NOT EXISTS nyayasetu;
