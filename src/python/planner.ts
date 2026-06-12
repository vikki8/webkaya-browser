/**
 * A tiny, deterministic natural-language → pandas planner.
 *
 * Its job is to make the local-data-analyst demo self-contained: it maps a
 * handful of question shapes onto pandas code with NO API key, so the value
 * prop ("agent code runs over your local data, in your browser") is
 * demonstrable offline. In production you swap this for a real LLM that emits
 * the same kind of Python snippet — the runner and governance are unchanged.
 */

export interface PlanResult {
  /** Python expression/statements to run; the final expression is the answer. */
  code: string;
  /** Human-readable description of what the plan does. */
  explanation: string;
  /** True when the planner fell back because it didn't recognise the question. */
  fallback: boolean;
}

function findColumn(question: string, columns: string[]): string | null {
  const lower = question.toLowerCase();
  // Prefer the longest matching column name to avoid partial collisions.
  const matches = columns
    .filter((c) => lower.includes(c.toLowerCase()))
    .sort((a, b) => b.length - a.length);
  return matches[0] ?? null;
}

function parseTopN(question: string): number {
  const match = question.match(/\btop\s+(\d+)/i) ?? question.match(/\b(\d+)\s+(?:rows|results)/i);
  return match ? Math.max(1, Math.min(1000, parseInt(match[1], 10))) : 5;
}

/**
 * Plan a question against a DataFrame named `df` with the given columns.
 * Recognises: row count, schema/describe, average/sum/min/max of a column,
 * group-by aggregation, and top-N preview. Falls back to `df.describe()`.
 */
export function planQuestion(question: string, columns: string[], df = 'df'): PlanResult {
  const q = question.toLowerCase().trim();
  const col = findColumn(question, columns);

  if (/\b(how many rows|row count|number of rows|count of rows)\b/.test(q)) {
    return { code: `len(${df})`, explanation: 'Count the rows.', fallback: false };
  }

  if (/\b(columns|schema|what columns|fields|structure)\b/.test(q)) {
    return { code: `list(${df}.columns)`, explanation: 'List the column names.', fallback: false };
  }

  const aggMatch = q.match(/\b(average|mean|sum|total|min|minimum|max|maximum|median)\b/);
  if (aggMatch && col) {
    const op = {
      average: 'mean', mean: 'mean', sum: 'sum', total: 'sum',
      min: 'min', minimum: 'min', max: 'max', maximum: 'max', median: 'median',
    }[aggMatch[1]] as string;
    // group-by form: "<agg> of <col> by <group>"
    const byMatch = q.match(/\bby\s+([a-z0-9_ ]+)$/i);
    const groupCol = byMatch ? findColumn(byMatch[1], columns) : null;
    if (groupCol && groupCol !== col) {
      return {
        code: `${df}.groupby(${JSON.stringify(groupCol)})[${JSON.stringify(col)}].${op}().to_dict()`,
        explanation: `${op} of "${col}" grouped by "${groupCol}".`,
        fallback: false,
      };
    }
    return {
      code: `${df}[${JSON.stringify(col)}].${op}()`,
      explanation: `${op} of column "${col}".`,
      fallback: false,
    };
  }

  if (/\b(top|first|head|preview|sample|show)\b/.test(q)) {
    const n = parseTopN(q);
    if (col && /\b(top|highest|largest)\b/.test(q)) {
      return {
        code: `${df}.nlargest(${n}, ${JSON.stringify(col)}).to_dict(orient="records")`,
        explanation: `Top ${n} rows by "${col}".`,
        fallback: false,
      };
    }
    return {
      code: `${df}.head(${n}).to_dict(orient="records")`,
      explanation: `First ${n} rows.`,
      fallback: false,
    };
  }

  return {
    code: `${df}.describe(include="all").to_dict()`,
    explanation: "Didn't recognise the question — showing a summary of every column.",
    fallback: true,
  };
}
