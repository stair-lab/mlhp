-- Protect cross-references from being treated as citations by section-bibliographies
-- This filter runs BEFORE section-bibliographies to prevent citeproc warnings

function Cite(el)
  for _, citation in ipairs(el.citations) do
    local id = citation.id
    -- Quarto cross-reference prefixes
    if id:match("^sec%-") or id:match("^eq%-") or id:match("^fig%-") or
       id:match("^tbl%-") or id:match("^alg%-") or id:match("^lst%-") or
       id:match("^thm%-") or id:match("^lem%-") or id:match("^cor%-") or
       id:match("^prp%-") or id:match("^cnj%-") or id:match("^def%-") or
       id:match("^exm%-") or id:match("^exr%-") then
      return {}
    end
  end
  return el
end
