import { useState, useEffect } from 'react';
import { Capabilities, detectCapabilities } from '../../engine/capability-detect';

export function useCapabilities() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    detectCapabilities().then(c => {
      setCaps(c);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return { caps, loading };
}
