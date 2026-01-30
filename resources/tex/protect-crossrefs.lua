-- Protect cross-references from being treated as citations by citeproc/section-bibliographies
-- This filter runs BEFORE section-bibliographies to prevent warnings like:
-- "Citeproc: citation sec-bradley-terry not found"

function Cite(el)
  -- Check if all citations in this element are actually cross-references
  local dominated_by_crossref = true
  for _, citation in ipairs(el.citations) do
    local id = citation.id
    -- Quarto cross-reference prefixes: sec-, eq-, fig-, tbl-, alg-, lst-, thm-, lem-, cor-, prp-, cnj-, def-, exm-, exr-
    if id:match("^sec%-") or id:match("^eq%-") or id:match("^fig%-") or
       id:match("^tbl%-") or id:match("^alg%-") or id:match("^lst%-") or
       id:match("^thm%-") or id:match("^lem%-") or id:match("^cor%-") or
       id:match("^prp%-") or id:match("^cnj%-") or id:match("^def%-") or
       id:match("^exm%-") or id:match("^exr%-") then
      -- This is a cross-reference, not a citation - skip it
    else
      dominated_by_crossref = false
    end
  end

  -- If all citations are cross-references, return empty to remove from citation processing
  -- Quarto will handle these separately as cross-references
  if dominated_by_crossref then
    return {}
  end

  return el
end
