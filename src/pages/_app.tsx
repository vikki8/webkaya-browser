import type { AppProps } from 'next/app';
import { useEffect } from 'react';
import '../styles/globals.css';
import { useStudioStore } from '../ui/store';

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    void useStudioStore.persist.rehydrate();
  }, []);

  return <Component {...pageProps} />;
}
